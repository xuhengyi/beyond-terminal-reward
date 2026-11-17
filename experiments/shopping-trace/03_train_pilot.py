"""Run an on-policy Shopping TRACE-GRPO pilot on PyTRIO."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import importlib
from itertools import chain, cycle
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import pytrio as trio


preflight = importlib.import_module("01_preflight")
from common import DEFAULT_SOURCE, normalized_messages, sha256, use_source, write_json
from tracking import Tracker

CONTEXT = 24576
MAX_OUTPUT = 512
GROUP_SIZE = 4
TASKS_PER_UPDATE = 2
MAX_SAMPLING_BATCHES = 30
TEMPERATURE = 0.7
TOP_P = 0.9
LEARNING_RATE = 1e-6
TURN_WEIGHT = 0.2
TRACE_PARAMETERS = {"epsilon": 0.1, "horizon": 3, "discount": 0.8, "terminal_weight": 2.0}


@dataclass
class ActionSample:
    message_index: int
    prompt: list[int]
    tokens: list[int]
    logprobs: list[float]


def prompt_tokens(tokenizer, messages, tools, *, generate):
    text = tokenizer.apply_chat_template(
        normalized_messages(messages), tools=tools, tokenize=False,
        add_generation_prompt=generate, enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


class RolloutPolicy:
    def __init__(self, sampler, tokenizer, seed):
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.seed = seed
        self.samples = []
        self.context_limit = False
        self.parse_error = False
        self.remote_error = None

    def complete(self, messages, tools):
        tokens = prompt_tokens(self.tokenizer, messages, tools, generate=True)
        if len(tokens) + MAX_OUTPUT > CONTEXT:
            self.context_limit = True
            return {"role": "assistant", "content": ""}
        try:
            result = self.sampler.sample(
                prompt=trio.ModelInput.from_ints(tokens), num_samples=1, return_text=True,
                sampling_params=trio.SamplingParams(
                    max_tokens=MAX_OUTPUT, temperature=TEMPERATURE, top_p=TOP_P,
                    stop=["<|im_end|>"], seed=self.seed + len(self.samples),
                ),
            ).result(timeout=180)
        except Exception as error:
            self.remote_error = {
                "code": getattr(error, "code", None),
                "http_status": getattr(error, "status_code", None),
                "type": type(error).__name__,
            }
            raise
        sequence = result.sequences[0]
        output = list(sequence.tokens)
        logprobs = list(sequence.logprobs)
        if len(output) != len(logprobs) or any(
            value is None or not math.isfinite(value) for value in logprobs
        ):
            raise ValueError("Action tokens and old logprobs are misaligned")
        self.samples.append(ActionSample(len(messages), tokens, output, logprobs))
        text = sequence.text if sequence.text is not None else self.tokenizer.decode(
            output, skip_special_tokens=False,
        )
        parse_call = importlib.import_module("03_check_action").parse_call
        try:
            name, arguments = parse_call(text, tools)
        except (ValueError, KeyError, TypeError):
            self.parse_error = True
            return {"role": "assistant", "content": text}
        body = re.sub(r"<tool_call>.*?</tool_call>|<\|im_end\|>", "", text, flags=re.DOTALL).strip()
        return {"role": "assistant", "content": body, "tool_calls": [{
            "id": f"trace_{self.seed}_{len(self.samples)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]}


def rollout(task, attempt, sampler, tokenizer, url, seed):
    from shopping_grpo.environment.client import ShopAgentEnv
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT, collect_for_task
    from shopping_grpo.training.grpo.trace import canonical_purchase_target

    class TraceEnvironment(ShopAgentEnv):
        def reset(self, task_id):
            self.include_trace_target = True
            state = super().reset(task_id)
            private = state.pop("_trace_target")
            self.target = canonical_purchase_target(private["asin"], private["options"])
            return state

    environment = TraceEnvironment(base_url=url)
    policy = RolloutPolicy(sampler, tokenizer, seed)
    spec = {"task_id": task["task_id"], "prompt": [{"role": "system", "content": SYSTEM_PROMPT}]}
    trajectory = collect_for_task(
        spec, policy, env_factory=lambda **kwargs: environment, base_url=url, max_steps=35,
        tools=SHOP_TOOL_SCHEMAS, attempt_index=attempt,
    )
    terminal = trajectory["terminal_result"]
    usable = bool(
        any(sample.tokens for sample in policy.samples)
        and not trajectory["error"] and not trajectory["release_error"]
        and terminal.get("reward_valid", True)
        and not terminal.get("reward_detail", {}).get("sampling_invalid", False)
    )
    outcome = terminal.get("termination_reason") or trajectory["status"]
    return {
        "task_id": task["task_id"], "attempt": attempt, "trajectory": trajectory,
        "samples": policy.samples, "target": getattr(environment, "target", None),
        "usable": usable, "outcome": outcome,
        "reward": float(terminal.get("reward", 0.0)) if trajectory["done"] else 0.0,
        "sample_input_tokens": sum(len(sample.prompt) for sample in policy.samples),
        "sample_output_tokens": sum(len(sample.tokens) for sample in policy.samples),
        "context_limit": policy.context_limit, "parse_error": policy.parse_error,
        "remote_error": policy.remote_error,
    }


def select_groups(executor, task_iterator, sampler, tokenizer, url, seed, directory, step):
    accepted = []
    attempts = []
    for batch in range(MAX_SAMPLING_BATCHES):
        tasks = [next(task_iterator) for _ in range(TASKS_PER_UPDATE)]
        groups = [
            [executor.submit(
                rollout, task, attempt, sampler, tokenizer, url,
                seed + step * 1000000 + batch * 10000 + group_index * 1000 + attempt * 100,
            ) for attempt in range(GROUP_SIZE)]
            for group_index, task in enumerate(tasks)
        ]
        for futures in groups:
            rows = [future.result() for future in futures]
            rewards = [row["reward"] for row in rows]
            keep = all(row["usable"] for row in rows) and max(rewards) - min(rewards) > 1e-8
            retained = keep and len(accepted) < TASKS_PER_UPDATE
            if retained:
                accepted.append(rows)
            attempts.append({
                "task_id": rows[0]["task_id"], "rewards": rewards,
                "usable": [row["usable"] for row in rows], "accepted": retained,
                "outcomes": [row["outcome"] for row in rows],
                "remote_errors": [row["remote_error"] for row in rows],
                "errors": [(
                    row["trajectory"].get("error") or row["trajectory"].get("release_error") or {}
                ).get("message") for row in rows],
                "sample_input_tokens": sum(row["sample_input_tokens"] for row in rows),
                "sample_output_tokens": sum(row["sample_output_tokens"] for row in rows),
                "context_limited_rollouts": sum(row["context_limit"] for row in rows),
                "parse_error_rollouts": sum(row["parse_error"] for row in rows),
            })
            write_json(directory / f"step-{step:03d}-sampling.json", attempts)
            print(f"step={step} task={rows[0]['task_id']} rewards={rewards} accepted={retained}", flush=True)
            for row in rows:
                error = row["remote_error"] or {}
                code = str(error.get("code") or "")
                if error.get("http_status") in {401, 402, 403} or code.startswith(("billing", "auth")):
                    raise RuntimeError(f"PyTRIO denied sampling: {code or error['http_status']}")
        if len(accepted) == TASKS_PER_UPDATE:
            return accepted, attempts
    raise RuntimeError(f"Step {step}: could not collect two usable groups in {MAX_SAMPLING_BATCHES} sampling batches")


def trace_prefixes(row, tokenizer):
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

    messages = row["trajectory"]["messages"]
    samples = row["samples"]
    initial = samples[0].prompt
    target = list(tokenizer.encode(row["target"], add_special_tokens=False))
    if not target or len(initial) + len(target) > CONTEXT:
        raise ValueError("Invalid TRACE target length")
    tail_budget = CONTEXT - len(initial) - len(target)
    scored = [initial]
    action_indexes = []
    for index, sample in enumerate(samples):
        position = sample.message_index
        if position + 1 < len(messages) and messages[position + 1]["role"] == "tool":
            prefix = prompt_tokens(tokenizer, messages[:position + 2], SHOP_TOOL_SCHEMAS, generate=False)
            if len(prefix) + len(target) > CONTEXT:
                shared = next(
                    (i for i, (a, b) in enumerate(zip(initial, prefix)) if a != b),
                    len(initial),
                )
                budget = tail_budget + len(initial) - shared
                prefix = prefix[:shared] + (prefix[-budget:] if budget else [])
            scored.append(prefix)
            action_indexes.append(index)
    return target, scored, action_indexes


def pack_samples(samples, advantages):
    """Coalesce only when the next sampled context preserves every prior token."""
    packed = []
    tokens, logprobs, values = [], [], []
    for sample, advantage in zip(samples, advantages, strict=True):
        if not sample.tokens:
            continue
        if tokens and sample.prompt[:len(tokens)] != tokens:
            packed.append((tokens, logprobs, values))
            tokens, logprobs, values = [], [], []
        missing = len(sample.prompt) - len(tokens)
        tokens.extend(sample.prompt[len(tokens):])
        logprobs.extend([None] * missing)
        values.extend([0.0] * missing)
        tokens.extend(sample.tokens)
        logprobs.extend(sample.logprobs)
        values.extend([advantage] * len(sample.tokens))
    if tokens:
        packed.append((tokens, logprobs, values))
    return packed


def make_datum(tokens, logprobs, advantages):
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray([
                token if logprob is not None else -100
                for token, logprob in zip(tokens[1:], logprobs[1:], strict=True)
            ], dtype=np.int64),
            "logprobs": np.asarray([
                0.0 if value is None else value for value in logprobs[1:]
            ], dtype=np.float32),
            "advantages": np.asarray(advantages[1:], dtype=np.float32),
        },
    )


def score_groups(executor, groups, make_reference, tokenizer):
    from pytrio.errors.control import ControlAPIError
    from shopping_grpo.training.grpo.trace import turn_rewards

    reference = make_reference()
    submitted = {}
    pending = []
    for group in groups:
        mean_reward = sum(row["reward"] for row in group) / len(group)
        for row in group:
            target, prefixes, action_indexes = trace_prefixes(row, tokenizer)
            keys = [(tuple(prefix), tuple(target)) for prefix in prefixes]
            for key in keys:
                if key not in submitted:
                    submitted[key] = executor.submit(preflight.score_target, reference, *key)
            pending.append((row, row["reward"] - mean_reward, action_indexes, keys))

    scores_by_key = {}
    expired_keys = []
    for key, future in submitted.items():
        try:
            scores_by_key[key] = future.result()
        except ControlAPIError as error:
            if error.code != "conflict" or error.status_code != 409:
                raise
            expired_keys.append(key)
    if expired_keys:
        print(f"Reference session conflict; refreshing {len(expired_keys)} pending scores", flush=True)
        reference = make_reference()
        retries = {key: executor.submit(preflight.score_target, reference, *key) for key in expired_keys}
        scores_by_key.update((key, future.result()) for key, future in retries.items())

    data = []
    details = []
    for row, outcome_advantage, indexes, keys in pending:
        scores = [scores_by_key[key]["mean_logprob"] for key in keys]
        credits = turn_rewards(scores, outcome_advantage=outcome_advantage, **TRACE_PARAMETERS)
        advantages = [outcome_advantage] * len(row["samples"])
        for index, credit in zip(indexes, credits, strict=True):
            advantages[index] += TURN_WEIGHT * credit
        packed = pack_samples(row["samples"], advantages)
        data.extend(make_datum(*item) for item in packed)
        details.append({
            "task_id": row["task_id"], "attempt": row["attempt"],
            "outcome": row["outcome"], "reward": row["reward"],
            "outcome_advantage": outcome_advantage,
            "advantages": advantages, "trace_credit": credits, "scored_action_indexes": indexes,
            "sign_flips": sum(
                value * outcome_advantage < 0 and bool(sample.tokens)
                for value, sample in zip(advantages, row["samples"], strict=True)
            ),
            "new_nonzero_actions": sum(
                outcome_advantage == 0 and value != 0 and bool(sample.tokens)
                for value, sample in zip(advantages, row["samples"], strict=True)
            ),
            "action_count": sum(bool(sample.tokens) for sample in row["samples"]),
            "action_tokens": row["sample_output_tokens"],
            "training_sequences": len(packed),
            "training_input_tokens": sum(len(tokens) - 1 for tokens, _, _ in packed),
        })
    return data, details, {
        "reference_requests": len(submitted),
        "reference_retry_requests": len(expired_keys),
        "reference_input_tokens": sum(score["scoring_input_tokens"] for score in scores_by_key.values()),
    }


def numeric_metrics(values):
    return {key: float(value) for key, value in values.items()
            if isinstance(value, (float, int)) and math.isfinite(value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3, metavar="1-100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=5704)
    parser.add_argument("--priority-tasks", type=int, nargs="*", default=[])
    parser.add_argument("--recovery-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--microbatch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--swanlab-mode", choices=["online", "local"], default="online")
    parser.add_argument("--swanlab-project", default="beyond-terminal-reward")
    args = parser.parse_args()
    backward_timeout = int(os.environ.get("TRACE_BACKWARD_TIMEOUT_SECONDS", "600"))
    submit_timeout = int(os.environ.get("TRACE_ACTOR_SUBMIT_TIMEOUT_SECONDS", "30"))
    if min(backward_timeout, submit_timeout) < 1:
        parser.error("TRACE timeouts must be positive")
    if not 1 <= args.updates <= 100:
        parser.error("--updates must be between 1 and 100")
    use_source(DEFAULT_SOURCE)
    from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset
    from shopping_grpo.evaluation.rollout import load_tasks

    source = DEFAULT_SOURCE / "data/grpo/train.jsonl"
    tasks = load_tasks(source)
    _, blind_ids = validate_canonical_blind_asset()
    if {row["task_id"] for row in tasks}.intersection(blind_ids):
        raise ValueError("GRPO training task IDs overlap Final-200")
    random.Random(args.seed).shuffle(tasks)
    by_id = {task["task_id"]: task for task in tasks}
    priorities = set(args.priority_tasks)
    if len(priorities) != len(args.priority_tasks) or not priorities <= by_id.keys():
        raise ValueError("Priority tasks must be unique IDs from the training split")
    tasks = [by_id[task_id] for task_id in args.priority_tasks] + [
        task for task in tasks if task["task_id"] not in priorities
    ]
    checkpoint = json.loads(args.run_result.read_text())
    model = checkpoint["base_model"]
    all_tasks = list(tasks)
    previous_steps = []
    previous_runs = []
    previous_pending = None
    if args.resume_from:
        previous = json.loads(args.resume_from.read_text())
        safe_status = previous["status"] == "completed" or (
            previous["status"] == "failed" and previous.get("failed_stage") in {"sampling", "scoring", "forward_backward"}
        )
        if (not safe_status
                or previous["base_model"] != model or previous["sft_report"] != str(args.run_result)
                or previous["training_tasks_sha256"] != sha256(source) or previous["seed"] != args.seed):
            raise ValueError("Previous pilot cannot be resumed from a confirmed checkpoint")
        previous_steps = previous["steps"]
        if (not previous_steps or len(previous_steps) >= args.updates
                or [item["step"] for item in previous_steps] != list(range(1, len(previous_steps) + 1))
                or not all(previous_steps[-1].get(field) for field in ("state_path", "sampler_path"))):
            raise ValueError("Resume requires consecutive saved updates and remaining steps")
        if previous.get("failed_stage") == "forward_backward":
            entry = previous.get("pending_update") or {}
            batch_file = args.resume_from.parent / f"step-{len(previous_steps) + 1:03d}-datums.json.gz"
            if entry.get("step") == len(previous_steps) + 1 and batch_file.is_file():
                previous_pending = (entry, batch_file)
        previous_runs = [*previous.get("previous_runs", []), str(args.resume_from)]
        attempted = {
            record["task_id"]
            for run in previous_runs
            for file in Path(run).parent.glob("step-*-sampling.json")
            for record in json.loads(file.read_text())
        }
        tasks = [task for task in tasks if task["task_id"] not in attempted]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    kind = "training" if args.updates > 5 else "pilot"
    directory = Path(__file__).resolve().parent / "outputs" / kind / run_id
    directory.mkdir(parents=True)
    args.run_name = f"trace-grpo-{kind}-{run_id}"
    report = {
        "status": "starting", "planned_updates": args.updates, "steps": previous_steps,
        "previous_runs": previous_runs, "max_sampling_batches_per_update": MAX_SAMPLING_BATCHES,
        "base_model": model, "sft_epoch": checkpoint["best_development_epoch"],
        "sft_report": str(args.run_result), "training_tasks_sha256": sha256(source),
        "seed": args.seed, "group_size": GROUP_SIZE, "tasks_per_update": TASKS_PER_UPDATE,
        "temperature": TEMPERATURE, "top_p": TOP_P, "learning_rate": LEARNING_RATE,
        "turn_weight": TURN_WEIGHT, "trace": TRACE_PARAMETERS,
        "reference": "frozen_base_model", "group_advantage": "reward_minus_group_mean",
        "runtime": "pytrio_ppo", "token_packing": "exact_prefix_only",
        "backward_timeout_seconds": backward_timeout,
        "actor_submit_timeout_seconds": submit_timeout,
        "context_labels": "ignore_index_-100", "priority_tasks": args.priority_tasks,
        "recovery_from": str(args.recovery_from) if args.recovery_from else None,
    }

    def save():
        destination = directory / "result.json"
        temporary = destination.with_suffix(".json.tmp")
        write_json(temporary, report)
        temporary.replace(destination)

    save()
    print(f"run report: {directory / 'result.json'}", flush=True)
    tracker = None
    try:
        tracker = Tracker(args, directory, {key: value for key, value in report.items() if key != "steps"})
        report["swanlab"] = tracker.metadata()
        report["status"] = "restoring"
        save()
        trio.configure(actor_event_submit_timeout_seconds=submit_timeout)
        service = importlib.import_module("02_sft").service(request_timeout_seconds=120)
        if previous_steps:
            latest = previous_steps[-1]
            training = service.create_training_client_from_state_with_optimizer(latest["state_path"])
            sampler_path = latest["sampler_path"]
        else:
            training = service.create_training_client_from_state(checkpoint["best_development_state_path"])
            sampler_path = checkpoint["best_development_sampler_path"]
        limits = training._limits
        requested = args.microbatch_size
        training._limits = replace(
            limits, max_batch_size=min(limits.max_batch_size or requested, requested),
        )
        report["server_training_limits"] = asdict(limits)
        report["training_limits"] = asdict(training._limits)
        save()
        def make_reference():
            return service.create_sampling_client(base_model=model)

        sampler = service.create_sampling_client(
            base_model=model, model_path=sampler_path,
        )
        tokenizer = sampler.get_tokenizer()
        local_environment = importlib.import_module("01_trajectory").local_environment
        with ThreadPoolExecutor(max_workers=8) as executor, local_environment(
            DEFAULT_SOURCE, args.port, directory, slots=8,
        ) as url:
            task_iterator = chain(tasks, cycle(all_tasks))
            for step in range(len(previous_steps) + 1, args.updates + 1):
                started = time.monotonic()
                if previous_pending is not None:
                    entry, batch_file = previous_pending
                    previous_pending = None
                    with gzip.open(batch_file, "rt") as handle:
                        data = [trio.Datum.model_validate(item) for item in json.load(handle)]
                    entry["reused_batch_from"] = str(batch_file)
                    usage, stats = entry["usage"], entry["stats"]
                    report["current_step"] = step
                    print(f"step={step} recovered prepared batch", flush=True)
                else:
                    report.update(status="sampling", current_step=step)
                    save()
                    groups, attempts = select_groups(
                        executor, task_iterator, sampler, tokenizer, url, args.seed, directory, step,
                    )
                    report["status"] = "scoring"
                    save()
                    print(f"step={step} scoring accepted trajectories", flush=True)
                    data, details, score_usage = score_groups(executor, groups, make_reference, tokenizer)
                    usage = {
                        key: sum(entry[key] for entry in attempts)
                        for key in ("sample_input_tokens", "sample_output_tokens")
                    }
                    usage.update(score_usage)
                    usage["training_input_tokens"] = sum(item["training_input_tokens"] for item in details)
                    stats = {
                        "sampled_trajectories": len(attempts) * GROUP_SIZE,
                        "used_trajectories": len(details),
                        "action_count": sum(item["action_count"] for item in details),
                        "sign_flips": sum(item["sign_flips"] for item in details),
                        "new_nonzero_actions": sum(item["new_nonzero_actions"] for item in details),
                        "valid_alternatives": sum(item["outcome"] == "valid_alternative_purchase" for item in details),
                        "mean_reward": sum(item["reward"] for item in details) / len(details),
                    }
                    entry = {"step": step, "usage": usage, "stats": stats, "rollouts": details}
                with gzip.open(directory / f"step-{step:03d}-datums.json.gz", "wt") as output:
                    json.dump([datum.model_dump(mode="json") for datum in data], output, separators=(",", ":"))
                report.update(status="forward_backward", pending_update=entry)
                save()
                print(f"step={step} backward datums={len(data)} usage={usage}", flush=True)
                forward = training.forward_backward(data, loss_fn="ppo").result(timeout=backward_timeout)
                loss = float(forward.metrics["loss_sum"])
                if not math.isfinite(loss):
                    raise ValueError("PPO returned a non-finite loss")
                adam = trio.AdamParams(
                    learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999,
                    eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0,
                )
                entry["loss"] = numeric_metrics(forward.metrics)
                report["status"] = "optimizer"
                save()
                update = training.optim_step(adam).result(timeout=240)
                entry.update(loss=numeric_metrics(forward.metrics),
                             optimizer=numeric_metrics(update.metrics or {}))
                report.pop("pending_update")
                report["steps"].append(entry)
                report["status"] = "checkpointing"
                save()
                label = f"trace-{run_id}-step{step}"
                entry["state_path"] = training.save_state(name=label + "-state").result(timeout=240).path
                save()
                entry["sampler_path"] = training.save_weights_for_sampler(name=label + "-sampler").result(timeout=240).path
                entry["seconds"] = time.monotonic() - started
                save()
                tracker.log({
                    **{f"tokens/{key}": value for key, value in usage.items()},
                    **{f"trace/{key}": value for key, value in stats.items()},
                    "train/ppo_loss_sum": loss, "train/step_seconds": entry["seconds"],
                }, step=step)
                print(f"step={step}/{args.updates} saved {usage} {stats}", flush=True)
                if step < args.updates:
                    sampler = training.create_sampling_client(entry["sampler_path"])
        report["status"] = "completed"
        save()
    except BaseException as error:
        report.update(failed_stage=report["status"], status="failed",
                      error_type=type(error).__name__, error_code=getattr(error, "code", None),
                      error_http_status=getattr(error, "status_code", None))
        save()
        if tracker is not None:
            tracker.finish(error)
        raise
    else:
        tracker.finish()


if __name__ == "__main__":
    main()
