"""购物学习的公共代码：完整轨迹、assistant mask 和 PyTRIO SFT 数据。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "vendor/shopping-grpo-longhorizon"
OUTPUTS = HERE / "outputs"


def use_source(source: Path) -> None:
    if not (source / "src/shopping_grpo/environment/client.py").is_file():
        raise ValueError(f"找不到购物源码；请用 --source-root 指定路径：{source}")
    sys.path.insert(0, str(source / "src"))


def split_path(source: Path, split: str) -> Path:
    return source.parents[1] / "artifacts/mvp/data" / f"sft-{split}.jsonl"


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"数据为空：{path}")
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("本学习实验要求每个 task_id 只有一条完整示范")
    for row in rows:
        validate_messages(row)
    return rows


def validate_messages(row: dict) -> None:
    """检查调用与返回一一对应，防止空正文的工具动作被漏掉。"""
    pending = None
    terminal = False
    for message in row["messages"]:
        role = message["role"]
        if role == "assistant":
            if pending is not None or terminal:
                raise ValueError("工具返回缺失，或终止动作后又出现模型动作")
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                raise ValueError("本实验要求每个 assistant 回合恰好调用一个工具")
            call = calls[0]
            pending = call["id"]
            arguments = call["function"]["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON object")
            terminal = call["function"]["name"] in {"buy_now", "finish_without_purchase"}
        elif role == "tool":
            if pending is None or message.get("tool_call_id") != pending:
                raise ValueError("tool_call_id 没有对应到前一轮模型动作")
            pending = None
        elif pending is not None:
            raise ValueError("工具返回前插入了其他消息")
    if pending is not None or not terminal or row["messages"][-1]["role"] != "tool":
        raise ValueError("示范缺少完整的终止动作和工具返回")


def normalized_messages(messages: list[dict]) -> list[dict]:
    """原始 OpenAI 数据的 arguments 是字符串；模板需要字典。"""
    result = deepcopy(messages)
    for message in result:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@dataclass
class EncodedTrajectory:
    task_id: int
    tokens: list[int]
    mask: list[int]
    spans: list[dict]

    @property
    def action_tokens(self) -> int:
        return sum(self.mask[1:])


def encode_trajectory(row: dict, tokenizer: Any, max_length: int) -> EncodedTrajectory:
    """先标注完整序列中的 assistant 区间，之后再做自回归右移。"""
    validate_messages(row)
    messages = normalized_messages(row["messages"])

    def render(part: list[dict], generation: bool = False) -> list[int]:
        text = tokenizer.apply_chat_template(
            part, tools=row["tools"], tokenize=False, add_generation_prompt=generation,
        )
        return list(tokenizer.encode(text, add_special_tokens=False))

    tokens = render(messages)
    if len(tokens) > max_length:
        raise ValueError(f"完整轨迹 {len(tokens)} tokens 超过 {max_length}；拒绝截断")
    mask = [0] * len(tokens)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = render(messages[:index], generation=True)
        through = render(messages[: index + 1])
        if tokens[:len(through)] != through:
            raise ValueError("chat template 改写了历史前缀，不能可靠对齐训练位置")
        start = 0
        for left, right in zip(prefix, through):
            if left != right:
                break
            start += 1
        end = len(through)
        if start == 0 or start >= end or any(mask[start:end]):
            raise ValueError("assistant token 边界无效或重叠")
        mask[start:end] = [1] * (end - start)
        spans.append({
            "message_index": index, "start": start, "end": end,
            "tool": message["tool_calls"][0]["function"]["name"],
        })
    if not sum(mask[1:]):
        raise ValueError("没有可监督的 assistant token")
    return EncodedTrajectory(int(row["task_id"]), tokens, mask, spans)


def shifted_fields(example: EncodedTrajectory, batch_size: int = 1) -> dict:
    """每条轨迹的权重和为 1/batch_size，形成按轨迹平均的 CE。"""
    if batch_size < 1 or not example.action_tokens:
        raise ValueError("无效 batch 或没有动作 token")
    denominator = example.action_tokens * batch_size
    return {
        "input_tokens": example.tokens[:-1],
        "target_tokens": example.tokens[1:],
        "weights": [value / denominator for value in example.mask[1:]],
    }


def to_datum(example: EncodedTrajectory, batch_size: int = 1):
    import numpy as np
    import pytrio as trio

    fields = shifted_fields(example, batch_size)
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(fields["input_tokens"]),
        loss_fn_inputs={
            "target_tokens": np.asarray(fields["target_tokens"], dtype=np.int64),
            "weights": np.asarray(fields["weights"], dtype=np.float32),
        },
    )
