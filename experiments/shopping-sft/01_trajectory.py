"""阶段一：查看真实训练示范，并可在本地 ShopSimulator 回放。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from common import DEFAULT_SOURCE, OUTPUTS, read_rows, split_path, use_source, write_json


@contextmanager
def local_environment(source: Path, port: int, output: Path, *, slots: int = 1):
    """只启动本次回放独占的本机服务；结束时关闭本次子进程。"""
    with socket.socket() as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise ValueError(f"端口 {port} 已占用；请指定另一个 --port")
    base = source / "environments/ShopSimulator"
    shop = base / "shop_env"
    python = base / ".venv-shopsim/bin/python"
    if not python.exists():
        raise ValueError("ShopSimulator 环境尚未安装，请先按购物仓库说明安装")
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "SHOP_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
        "SHOP_ENV_CONFIG": str(shop / "configs/environment.json"),
        "SHOP_SEARCH_INDEX": str(shop / "search_engine/products.sqlite3"),
        "SHOP_MAX_STEPS": "35", "SHOPSIM_ENV_SLOTS": str(slots),
    }
    code = (
        "import pack_api; pack_api.initialize_environments(); "
        f"pack_api.app.run(host='127.0.0.1', port={port}, use_reloader=False)"
    )
    with (output / "environment.log").open("w") as log:
        process = subprocess.Popen(
            [str(python), "-u", "-c", code], cwd=shop / "shop_env",
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("环境启动失败，详见 outputs/environment.log")
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.2)
            else:
                raise TimeoutError("本地环境 60 秒内未就绪")
            yield f"http://127.0.0.1:{port}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def actions(row: dict):
    for index, message in enumerate(row["messages"]):
        if message["role"] == "assistant":
            call = message["tool_calls"][0]
            function = call["function"]
            parameters = function["arguments"]
            if isinstance(parameters, str):
                parameters = json.loads(parameters)
            yield index, message, function["name"], parameters


def replay(row: dict, source: Path, url: str) -> dict:
    use_source(source)
    from shopping_grpo.environment.actions import action_reject_reason
    from shopping_grpo.environment.client import ShopAgentEnv
    from shopping_grpo.environment.observation import render_structured_observation
    from shopping_grpo.environment.tools import tool_call_to_action

    records = []
    with ShopAgentEnv(base_url=url) as env:
        reset = env.reset(row["task_id"])
        observation = render_structured_observation(reset["observation_state"])
        for turn, (index, _, name, parameters) in enumerate(actions(row), start=1):
            reason = action_reject_reason(name, parameters, observation)
            if reason:
                raise ValueError(f"第 {turn} 轮动作守卫拒绝：{reason}")
            action = tool_call_to_action(name, parameters)
            result = env.step(action)
            records.append({"turn": turn, "message_index": index, "tool": name,
                            "arguments": parameters, "action": action, "result": result})
            observation = render_structured_observation(result["observation_state"])
            if env.done:
                if turn != sum(1 for _ in actions(row)):
                    raise ValueError("回放提前终止，与示范动作序列不符")
                break
        if not env.done:
            raise ValueError("示范回放后环境没有终止")
    return {"task_id": row["task_id"], "kind": "teacher_trajectory_replay",
            "is_model_rollout": False, "reset": reset, "steps": records,
            "terminal": records[-1]["result"]}


def write_lesson(row: dict, output: Path, replay_result: dict | None) -> Path:
    query = next(m["content"] for m in row["messages"] if m["role"] == "user")
    lines = ["# 第一课：一条真实购物示范", "", f"任务 ID：`{row['task_id']}`", "",
             query, "", "## 跟着动作读", "",
             "| 轮次 | 工具 | 参数 | 教师当时的说明 |", "|---|---|---|---|"]
    for turn, (_, message, name, parameters) in enumerate(actions(row), start=1):
        note = str(message.get("content") or "（正文为空，工具调用仍是训练目标）")
        note = note.replace("|", "\\|").replace("\n", " ")
        args = json.dumps(parameters, ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {turn} | `{name}` | `{args}` | {note} |")
    lines += ["", "## 先回答这四个问题", "",
              "1. 用户的品类、预算、材质和规格要求分别是什么？",
              "2. 哪次工具返回支持了最终商品及规格的选择？请查看 example.json 的对应 tool 消息。",
              "3. 为什么从信息子页返回后才能选择规格？当前页面如何限制合法动作？",
              "4. 哪些 assistant 轮次的正文为空？为什么仍应监督其中的 tool_calls？", "",
              "## 与 SFT 的连接", "",
              "system/user/tool 是上下文；assistant 的正文、工具名、参数和结束标记是监督目标。",
              "SFT 学习整条验收过的示范，不代表示范每一步都是最优。额外搜索是否必要，可以留作后续失败分析。", ""]
    if replay_result:
        final = replay_result["terminal"]
        scoring = {k: v for k, v in final.items() if k != "observation"}
        lines += ["## 本次环境回放", "",
                  "以下是固定教师动作在真实环境中的回放结果，用于检查环境与奖励；不计作模型能力或训练收益。", "",
                  "```json", json.dumps(scoring, ensure_ascii=False, indent=2), "```", "",
                  "评分结果只保存在回放报告中，不会追加进 SFT 的模型输入。", ""]
    path = output / "lesson_01.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--port", type=int, default=5701)
    args = parser.parse_args()
    rows = read_rows(args.data or split_path(args.source_root, "train"))
    row = next((r for r in rows if r["task_id"] == args.task_id), None) if args.task_id else rows[0]
    if row is None:
        raise ValueError("指定任务不在训练示范中")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "example.json", row)
    result = None
    if args.replay:
        with local_environment(args.source_root, args.port, args.output_dir) as url:
            result = replay(row, args.source_root, url)
        write_json(args.output_dir / "replay.json", result)
    path = write_lesson(row, args.output_dir, result)
    print(f"任务 {row['task_id']}，{sum(1 for _ in actions(row))} 轮工具交互")
    print(f"学习讲解：{path}")
    if result:
        print("已完成真实环境回放；这不是模型自主执行结果。")


if __name__ == "__main__":
    main()
