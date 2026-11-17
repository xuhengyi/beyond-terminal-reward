"""完整购物评测：开发题试跑，或显式执行冻结Final200的SFT/Base/TRACE-GRPO对照。"""

import argparse
from copy import deepcopy
import hashlib
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import shutil
import time

from common import DEFAULT_SOURCE, OUTPUTS, normalized_messages, read_rows, use_source, write_json


class PilotBudgetExceeded(ValueError):
    pass


class ReplayMismatchError(RuntimeError):
    pass


def is_model_error(error):
    return (error or {}).get("type") in {"ValueError", "JSONDecodeError", "KeyError"}


class SamplingPolicy:
    def __init__(self, sampler, tokenizer, directory, budget_yuan=5):
        self.sampler, self.tokenizer, self.directory = sampler, tokenizer, directory
        self.budget_yuan = budget_yuan
        self.calls = []
        self.task_id = None
        self.last_context_tokens = None
        self.replay = None
        failures_path = directory / 'request_failures.json'
        self.request_failures = json.loads(failures_path.read_text()) if failures_path.exists() else []

    def sample_with_retry(self, tokens):
        import pytrio as trio
        from pytrio.errors.actor import ActorAPIError
        max_request_cost = (len(tokens)*1.49 + 512*4.54)/1e6
        for attempt in range(1, 4):
            reserved = sum(x['billing_uncertainty_reserve_yuan'] for x in self.request_failures)
            spent = sum(c['cost_no_cache_yuan'] for c in self.calls)
            if spent + reserved + max_request_cost > self.budget_yuan:
                raise PilotBudgetExceeded('评测及失败请求的保守费用预留达到预算阈值')
            try:
                return self.sampler.sample(prompt=trio.ModelInput.from_ints(tokens), num_samples=1,
                    sampling_params=trio.SamplingParams(max_tokens=512, temperature=0,
                                                       stop=['<|im_end|>']), return_text=True).result()
            except ActorAPIError as error:
                if error.status_code != 499:
                    raise
                self.request_failures.append({
                    'task_id': self.task_id, 'attempt': attempt, 'http_status':499,
                    'time_utc':datetime.now(timezone.utc).isoformat(),
                    'request_id':getattr(error,'request_id',''),
                    'input_tokens':len(tokens), 'output_tokens':None,
                    'billing_uncertainty_reserve_yuan':max_request_cost,
                    'billing_status':'unknown_not_added_to_known_token_cost',
                })
                write_json(self.directory/'request_failures.json', self.request_failures)
                print(f'task={self.task_id} HTTP499 attempt={attempt}/3', flush=True)
                if attempt == 3:
                    raise
                time.sleep(5*attempt)

    def start_replay(self, trajectory):
        history = trajectory["messages"]
        indices = [i for i, m in enumerate(history) if m["role"] == "assistant"]
        calls = [c for c in self.calls if c["task_id"] == trajectory["task_id"]]
        if len(indices) != len(calls):
            raise ReplayMismatchError("中断任务的已返回调用数与轨迹不一致，停止恢复")
        self.replay = {"history": history, "indices": indices, "calls": calls, "position": 0}

    def replay_response(self, messages):
        replay = self.replay
        if replay is None:
            return None
        position = replay["position"]
        if position == len(replay["indices"]):
            if messages != replay["history"]:
                raise ReplayMismatchError("本地回放后的观察与中断时不一致，未发出新模型请求")
            self.replay = None
            print(f"task={self.task_id} replay_verified={position} continuing_remote_sampling", flush=True)
            return None
        index = replay["indices"][position]
        if messages != replay["history"][:index]:
            raise ReplayMismatchError("回放前缀与原轨迹不一致，未发出新模型请求")
        self.last_context_tokens = replay["calls"][position]["input_tokens_sent"]
        replay["position"] += 1
        print(f"task={self.task_id} replay={position+1}/{len(replay['indices'])} model_request=False", flush=True)
        return deepcopy(replay["history"][index])

    def complete(self, messages, tools):
        cached = self.replay_response(messages)
        if cached is not None:
            return cached
        parse_call = importlib.import_module("03_check_action").parse_call
        prompt = self.tokenizer.apply_chat_template(normalized_messages(messages), tools=tools,
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        if len(tokens) + 512 > 32768:
            raise ValueError("完整上下文超过32768，拒绝静默截断")
        estimated = sum(c["cost_no_cache_yuan"] for c in self.calls)
        if estimated + (len(tokens)*1.49 + 512*4.54)/1e6 > self.budget_yuan:
            raise PilotBudgetExceeded("评测达到推理估算预算上限")
        started = time.monotonic()
        result = self.sample_with_retry(tokens)
        seq = result.sequences[0]
        text = seq.text if seq.text is not None else self.tokenizer.decode(seq.tokens, skip_special_tokens=False)
        record = {"task_id": self.task_id, "input_tokens_sent": len(tokens),
                  "output_tokens": len(seq.tokens), "server_input_tokens": result.input_tokens,
                  "server_output_tokens": result.output_tokens,
                  "seconds": time.monotonic()-started, "stop_reason": seq.stop_reason,
                  "assistant_text": text,
                  "cost_no_cache_yuan": (len(tokens)*1.49 + len(seq.tokens)*4.54)/1e6}
        self.calls.append(record)
        write_json(self.directory / "calls.json", self.calls)
        self.last_context_tokens = len(tokens)
        print(f"task={self.task_id} call={len(self.calls)} input={len(tokens)} output={len(seq.tokens)} seconds={record['seconds']:.1f}", flush=True)
        if "<tool_call>" not in text:
            return {"role": "assistant", "content": text}
        name, arguments = parse_call(text, tools)
        body = re.sub(r"<tool_call>.*?</tool_call>|<\|im_end\|>", "", text, flags=re.DOTALL).strip()
        return {"role": "assistant", "content": body, "tool_calls": [{
            "id": f"pilot_{self.task_id}_{len(self.calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--split", choices=["development", "final"], default="development")
    parser.add_argument("--allow-blind-final", action="store_true")
    parser.add_argument("--model-kind", choices=["sft", "base", "trace-grpo"], default="sft")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--budget-yuan", type=float, default=5)
    parser.add_argument("--port", type=int, default=5701)
    parser.add_argument("--resume-from", type=Path, help="原评测目录；只恢复基础设施中断，不重跑模型失败题")
    args = parser.parse_args()
    use_source(args.source_root)
    from shopping_grpo.evaluation.rollout import collect_for_task, SYSTEM_PROMPT
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    from shopping_grpo.evaluation.summary import summarize_trajectories
    from shopping_grpo.evaluation.blind_guard import guard_blind_final
    if args.split == "final":
        if not args.allow_blind_final or args.tasks != 200:
            parser.error("正式测试必须显式 --allow-blind-final --tasks 200，保持固定分母")
        source = args.source_root / "data/evaluation/tasks.jsonl"
        guard_blind_final([source], allowed=True)
        from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset
        guard, canonical_ids = validate_canonical_blind_asset()
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(rows) != 200 or {int(r['task_id']) for r in rows} != canonical_ids:
            raise ValueError("正式测试任务与固定Final-200不一致")
        if hashlib.sha256(source.read_bytes()).hexdigest() != guard['task_sha256']:
            raise ValueError("正式测试任务文件哈希与上游冻结资产不一致")
    else:
        if not 1 <= args.tasks <= 10:
            parser.error("开发试跑限定1–10题")
        source = args.source_root / "data/sft/validation.jsonl"
        guard_blind_final([source], allowed=False)
        rows = read_rows(source)[:args.tasks]
    checkpoint = json.loads(args.run_result.read_text())
    if args.model_kind == "trace-grpo":
        steps = checkpoint["steps"]
        if (
            checkpoint["status"] != "completed"
            or checkpoint["current_step"] != 100
            or [step["step"] for step in steps] != list(range(1, 101))
            or not steps[-1].get("sampler_path")
        ):
            parser.error("TRACE-GRPO evaluation requires a completed step 100 sampler checkpoint")
        weights = steps[-1]["sampler_path"]
    else:
        weights = checkpoint["best_development_sampler_path"] if args.model_kind == "sft" else None
    folder = "evaluation_final" if args.split == "final" else "evaluation_pilots"
    directory = args.resume_from or args.output_dir / folder / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not args.resume_from:
        directory.mkdir(parents=True)
    report = {"kind": "complete_shopping_evaluation", "split": args.split, "model_kind": args.model_kind,
              "base_model": checkpoint["base_model"], "weights": weights,
              "task_ids": [r["task_id"] for r in rows], "max_steps": 35, "max_output_tokens": 512,
              "temperature": 0, "concurrency": 1, "status": "starting", "results": [],
              "task_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
              "tools_sha256": hashlib.sha256(json.dumps(SHOP_TOOL_SCHEMAS,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
              "budget_no_cache_estimated_yuan": args.budget_yuan}
    trajectories = []
    old_calls = []
    interrupted = None
    prior_wall_seconds = 0
    if args.resume_from:
        old = json.loads((directory / "result.json").read_text())
        if old["status"] != "stopped_for_inspection":
            raise ValueError("只允许恢复已明确停止的任务，避免重复运行")
        keys = ["split", "model_kind", "base_model", "weights", "task_ids", "max_steps", "max_output_tokens",
                "temperature", "concurrency", "task_file_sha256", "system_prompt_sha256", "tools_sha256",
                "budget_no_cache_estimated_yuan"]
        if any(old[k] != report[k] for k in keys):
            raise ValueError("恢复配置与原评测不一致，拒绝改变协议或预算")
        previous = old["results"]
        if [r['task_id'] for r in previous] != report['task_ids'][:len(previous)]:
            raise ValueError("既有结果不是固定任务顺序的前缀")
        if previous and previous[-1].get("error"):
            last = previous[-1]
            if is_model_error(last['error']) or last['error']['type'] == 'PilotBudgetExceeded':
                raise ValueError("不重跑模型失败或预算触顶的任务")
            interrupted = json.loads((directory / f"task_{last['task_id']}.json").read_text())
            if interrupted.get('release_error'):
                raise ValueError("环境释放异常，需先单独修复")
            previous = previous[:-1]
        archive = directory / "resume_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for name in ['result.json', 'calls.json']:
            shutil.copy2(directory / name, archive / name)
        if interrupted:
            shutil.copy2(directory / f"task_{interrupted['task_id']}.json", archive / f"task_{interrupted['task_id']}.json")
        if (directory / 'swanlab_sync/swanlab.json').exists():
            shutil.copy2(directory / 'swanlab_sync/swanlab.json', archive / 'swanlab.json')
        trajectories = [json.loads((directory / f"task_{r['task_id']}.json").read_text()) for r in previous]
        old_calls = json.loads((directory / 'calls.json').read_text())
        prior_wall_seconds = old.get('wall_seconds', 0)
        report = old
        report['results'] = previous
        report['status'] = 'resuming'
        report['resume_history'] = report.get('resume_history', []) + [{
            'archive': str(archive), 'retained_tasks': len(previous),
            'interrupted_task': interrupted['task_id'] if interrupted else None,
            'retained_returned_calls': len(old_calls), 'method': 'local_replay_then_continue',
            'sampling_499_attempts':3, 'generic_request_timeout_seconds':120,
            'note':'sampling使用SDK独立超时；通用请求超时不替代sampling超时'}]
        report['summary'] = summarize_trajectories(report['task_ids'], trajectories)
    write_json(directory / "result.json", report)
    service = importlib.import_module("02_sft").service(request_timeout_seconds=120)
    sampler = service.create_sampling_client(base_model=checkpoint["base_model"], model_path=weights)
    policy = SamplingPolicy(sampler, sampler.get_tokenizer(), directory, args.budget_yuan)
    policy.calls = old_calls
    environment = importlib.import_module("01_trajectory").local_environment
    wall_start = time.monotonic()
    with environment(args.source_root, args.port, directory) as url:
        for row in rows[len(trajectories):]:
            policy.task_id = row["task_id"]
            replaying = interrupted is not None and row['task_id'] == interrupted['task_id']
            if replaying:
                policy.start_replay(interrupted)
            started = time.monotonic()
            task = {"task_id": row["task_id"], "prompt": row["messages"][:1] if args.split == "development" else [{"role":"system","content":SYSTEM_PROMPT}]}
            tools = row["tools"] if args.split == "development" else SHOP_TOOL_SCHEMAS
            trajectory = collect_for_task(task, policy, base_url=url, max_steps=35, tools=tools)
            trajectory["wall_seconds"] = time.monotonic()-started
            if replaying:
                trajectory['resume_prior_wall_seconds'] = interrupted['wall_seconds']
                trajectory['wall_seconds'] += interrupted['wall_seconds']
                trajectory['resumed_from_infrastructure_error'] = interrupted['error']['type']
            trajectories.append(trajectory)
            write_json(directory / f"task_{row['task_id']}.json", trajectory)
            result = {"task_id": row["task_id"], "status": trajectory["status"],
                      "steps": len(trajectory["steps"]), "reward": trajectory["final_reward"],
                      "seconds": trajectory["wall_seconds"], "error": trajectory.get("error")}
            report["results"].append(result)
            report["status"] = "running"
            report["summary"] = summarize_trajectories(report["task_ids"], trajectories)
            report["input_tokens_sent"] = sum(c["input_tokens_sent"] for c in policy.calls)
            report["output_tokens"] = sum(c["output_tokens"] for c in policy.calls)
            report["cost_no_cache_yuan"] = sum(c["cost_no_cache_yuan"] for c in policy.calls)
            write_json(directory / "result.json", report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            model_error = is_model_error(result["error"])
            if (result["error"] and not model_error) or trajectory.get("release_error"):
                report["status"] = "stopped_for_inspection"
                break
    report["wall_seconds"] = prior_wall_seconds + time.monotonic()-wall_start
    report["summary"] = summarize_trajectories(report["task_ids"], trajectories)
    report["input_tokens_sent"] = sum(c["input_tokens_sent"] for c in policy.calls)
    report["output_tokens"] = sum(c["output_tokens"] for c in policy.calls)
    report["cost_no_cache_yuan"] = sum(c["cost_no_cache_yuan"] for c in policy.calls)
    n = len(trajectories)
    if n:
        report["estimate_200_tasks"] = {
            "cost_no_cache_yuan": report["cost_no_cache_yuan"] / n * 200,
            "serial_minutes": sum(t["wall_seconds"] for t in trajectories) / n * 200 / 60,
            "basis_tasks": n, "is_actual_bill": False}
    if len(trajectories) == args.tasks and report["status"] != "stopped_for_inspection":
        report["status"] = "completed"
    write_json(directory / "result.json", report)
    print(f"评测报告：{directory / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
