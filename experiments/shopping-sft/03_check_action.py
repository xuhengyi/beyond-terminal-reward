"""SFT 后的最小开发检查：让模型自己生成首个工具调用，并由环境真实执行。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re

from common import DEFAULT_SOURCE, OUTPUTS, normalized_messages, read_rows, split_path, use_source, write_json


def parse_call(text: str, tools: list[dict]) -> tuple[str, dict]:
    """支持 Qwen XML 和 JSON 工具格式；只接受一个符合公开 schema 的调用。"""
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if len(blocks) != 1 or text.count("<tool_call>") != 1:
        raise ValueError("模型必须生成恰好一个完整 tool_call")
    body = blocks[0]
    if body.lstrip().startswith("{"):
        payload = json.loads(body)
        name, arguments = payload["name"], payload["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    else:
        match = re.fullmatch(r"<function=([^>]+)>\s*(.*?)\s*</function>", body, re.DOTALL)
        if not match:
            raise ValueError("不完整的 Qwen function 格式")
        name, parameters = match.groups()
        matches = list(re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", parameters, re.DOTALL))
        if re.sub(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", "", parameters, flags=re.DOTALL).strip():
            raise ValueError("参数标签外含有多余内容")
        arguments = {m.group(1): m.group(2).strip() for m in matches}
        if len(arguments) != len(matches):
            raise ValueError("存在重复参数")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    if name not in schemas or name == "think" or not isinstance(arguments, dict):
        raise ValueError("工具名或参数对象不合法")
    schema = schemas[name]
    if set(arguments) - set(schema.get("properties", {})):
        raise ValueError("包含工具 schema 外的参数")
    if set(schema.get("required", [])) - set(arguments):
        raise ValueError("缺少工具必需参数")
    for key, value in arguments.items():
        expected = schema["properties"][key]
        if expected.get("type") == "string" and not isinstance(value, str):
            raise ValueError("工具参数类型错误")
        if "enum" in expected and value not in expected["enum"]:
            raise ValueError("工具参数不在允许的枚举中")
    return name, arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--weights", help="save_weights_for_sampler 返回的路径；省略则检查 Base")
    parser.add_argument("--run-result", type=Path, help="读取 02_sft.py 的 result.json，自动取得模型和采样权重")
    parser.add_argument("--tasks", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=5701)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    if args.run_result:
        previous = json.loads(args.run_result.read_text())
        args.base_model = previous["base_model"]
        args.weights = previous.get("best_development_sampler_path") or previous["sampler_path"]
    # 数字开头的脚本文件不能用于普通 import 语句，使用 importlib 复用环境及服务函数。
    local_environment = importlib.import_module("01_trajectory").local_environment
    service = importlib.import_module("02_sft").service
    client = service()
    sampler = client.create_sampling_client(base_model=args.base_model, model_path=args.weights)
    tokenizer = sampler.get_tokenizer()
    import pytrio as trio
    use_source(args.source_root)
    from shopping_grpo.environment.actions import action_reject_reason
    from shopping_grpo.environment.client import ShopAgentEnv
    from shopping_grpo.environment.observation import render_structured_observation
    from shopping_grpo.environment.tools import tool_call_to_action

    rows = read_rows(split_path(args.source_root, "validation"))[:args.tasks]
    report = {"kind": "model_first_action_development_check", "base_model": args.base_model,
              "weights": args.weights, "results": [], "is_full_task_evaluation": False}
    run = args.output_dir / "action_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    with local_environment(args.source_root, args.port, run) as url:
        for row in rows:
            entry = {"task_id": row["task_id"], "legal_tool_executed": False}
            report["results"].append(entry)
            with ShopAgentEnv(base_url=url) as env:
                reset = env.reset(row["task_id"])
                visible = render_structured_observation(reset["observation_state"])
                # 只使用与训练相同的 system/user 和公开工具；不把 reset 原始评分字段送给模型。
                prompt = tokenizer.apply_chat_template(
                    normalized_messages(row["messages"][:2]), tools=row["tools"],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                result = sampler.sample(
                    prompt=trio.ModelInput.from_ints(prompt_tokens), num_samples=1,
                    sampling_params=trio.SamplingParams(max_tokens=args.max_tokens, temperature=0, stop=["<|im_end|>"]),
                    return_text=True,
                ).result()
                sequence = result.sequences[0]
                text = sequence.text if sequence.text is not None else tokenizer.decode(sequence.tokens, skip_special_tokens=False)
                entry.update({"assistant_text": text, "prompt_tokens": len(prompt_tokens),
                              "completion_tokens": list(sequence.tokens)})
                try:
                    name, arguments = parse_call(text, row["tools"])
                    reason = action_reject_reason(name, arguments, visible)
                    if reason:
                        raise ValueError(reason)
                    executed = env.step(tool_call_to_action(name, arguments))
                    entry.update({"tool": name, "arguments": arguments, "legal_tool_executed": True,
                                  "public_observation": render_structured_observation(executed["observation_state"])})
                except (ValueError, KeyError) as error:
                    entry["action_error"] = str(error)
            write_json(run / "result.json", report)
    print(f"合法首轮调用：{sum(x['legal_tool_executed'] for x in report['results'])}/{len(rows)}")
    print(f"开发检查：{run / 'result.json'}")
    print("这里只检查模型首轮动作能否真实执行，不表示购物任务已经成功。")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
