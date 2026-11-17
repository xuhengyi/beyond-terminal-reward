"""只把已有评测JSON同步到SwanLab，不调用模型、不修改评测结果。"""

import argparse
import json
from pathlib import Path
import time

from tracking import Tracker


def task_metrics(report, calls, completed):
    rows = report["results"][:completed]
    ids = {r["task_id"] for r in rows}
    used = [c for c in calls if c["task_id"] in ids]
    success_ids = set(report.get("summary", {}).get("strict_success_task_ids", []))
    successes = len(ids & success_ids)
    last = rows[-1]
    total = len(report["task_ids"])
    return {
        "eval/completed_tasks": completed,
        "eval/total_tasks": total,
        "eval/coverage": completed / total,
        "eval/strict_successes": successes,
        "eval/success_rate_completed": successes / completed,
        "eval/strict_success_rate_fixed_denominator": successes / total,
        "eval/input_tokens": sum(c["input_tokens_sent"] for c in used),
        "eval/output_tokens": sum(c["output_tokens"] for c in used),
        "eval/cost_no_cache_estimated_yuan": sum(c["cost_no_cache_yuan"] for c in used),
        "eval/task_seconds_total": sum(r["seconds"] for r in rows),
        "task/id": last["task_id"],
        "task/reward": last["reward"],
        "task/steps": last["steps"],
        "task/seconds": last["seconds"],
        "task/strict_success": int(last["task_id"] in success_ids),
    }


def read_snapshot(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # 原评测写JSON时可能有极短暂的未写完窗口，等待下次读取。
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--swanlab-mode", default="online", choices=["online", "local"])
    parser.add_argument("--swanlab-project", default="beyond-terminal-reward")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    report = read_snapshot(args.result)
    if report is None:
        raise ValueError("评测报告尚不可读，请稍后重试")
    args.run_name = args.run_name or f"eval-{report.get('model_kind','sft')}-{report['split']}-{args.result.parent.name}"
    config = {k: report.get(k) for k in ["kind", "split", "model_kind", "base_model", "weights",
              "max_steps", "max_output_tokens", "temperature", "concurrency", "task_file_sha256",
              "system_prompt_sha256", "tools_sha256"]}
    config.update(source_result=str(args.result.resolve()), total_tasks=len(report["task_ids"]),
                  cost_is_actual_bill=False, source="existing_evaluation_records")
    tracker = Tracker(args, args.result.parent / "swanlab_sync", config)
    uploaded = 0
    print(json.dumps(tracker.metadata(), ensure_ascii=False), flush=True)
    try:
        while True:
            report = read_snapshot(args.result)
            calls = read_snapshot(args.result.parent / "calls.json")
            if report is None or calls is None:
                time.sleep(2)
                continue
            for n in range(uploaded + 1, len(report["results"]) + 1):
                tracker.log(task_metrics(report, calls, n), step=n)
                uploaded = n
            if uploaded:
                print(f"uploaded_tasks={uploaded}/{len(report['task_ids'])} status={report['status']}", flush=True)
            if report["status"] == "completed":
                tracker.log({"checks/evaluation_completed": 1}, step=uploaded)
                tracker.finish()
                return
            if report["status"] == "stopped_for_inspection":
                tracker.finish(RuntimeError("evaluation_stopped_for_inspection"))
                return
            time.sleep(5)
    except BaseException as error:
        tracker.finish(error)
        raise


if __name__ == "__main__":
    main()
