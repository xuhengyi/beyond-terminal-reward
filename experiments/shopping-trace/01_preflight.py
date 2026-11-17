"""Check the shopping TRACE rollout and frozen-reference scoring path."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pytrio as trio


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "experiments/shopping-sft"))

from common import DEFAULT_SOURCE, normalized_messages, use_source, write_json


def score_target(reference, prefix, target):
    tokens = [*prefix, *target]
    start = time.monotonic()
    logprobs = reference.compute_logprobs(trio.ModelInput.from_ints(tokens)).result(timeout=180)
    values = logprobs[len(prefix):]
    if len(values) != len(target) or any(
        value is None or not math.isfinite(value) for value in values
    ):
        raise ValueError("TRACE target logprobs did not align with the scored target")
    return {
        "prefix_tokens": len(prefix),
        "target_tokens": len(target),
        "scoring_input_tokens": len(tokens),
        "mean_logprob": sum(values) / len(values),
        "seconds": time.monotonic() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--check-training", action="store_true")
    parser.add_argument("--check-backward", action="store_true")
    parser.add_argument("--port", type=int, default=5702)
    args = parser.parse_args()
    use_source(args.source_root)
    from shopping_grpo.environment.client import ShopAgentEnv
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    from shopping_grpo.evaluation.blind_guard import guard_blind_final
    from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT, collect_for_task, load_tasks
    from shopping_grpo.training.grpo.trace import canonical_purchase_target, turn_rewards

    data = args.source_root / "data/grpo/validation.jsonl"
    guard_blind_final([data], allowed=False)
    tasks = load_tasks(data)
    row = next((item for item in tasks if item["task_id"] == args.task_id), None) if args.task_id else tasks[0]
    if row is None:
        parser.error("task id not found in the GRPO development split")
    task_id = row["task_id"]
    checkpoint = json.loads(args.run_result.read_text())
    model = checkpoint["base_model"]
    weights = checkpoint["best_development_sampler_path"]
    if not weights:
        raise ValueError("SFT checkpoint sampler path is missing")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(__file__).resolve().parent / "outputs/preflight" / run_id
    directory.mkdir(parents=True)
    report = {
        "status": "starting",
        "task_id": task_id,
        "base_model": model,
        "policy_checkpoint": str(args.run_result),
        "sft_epoch": checkpoint["best_development_epoch"],
        "reference": "frozen_base_model",
        "parameter_updates": 0,
    }

    def stage(name, **fields):
        report.update(status=name, **fields)
        write_json(directory / "result.json", report)
        print(f"{name}: {directory / 'result.json'}", flush=True)

    class TraceProbeEnv(ShopAgentEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.include_trace_target = True

        def reset(self, task_id):
            result = super().reset(task_id)
            target = result.pop("_trace_target")
            self.target = canonical_purchase_target(target["asin"], target["options"])
            return result

    sample_policy = importlib.import_module("04_evaluate_pilot").SamplingPolicy

    class ProbePolicy(sample_policy):
        def sample_with_retry(self, tokens):
            result = super().sample_with_retry(tokens)
            sequence = result.sequences[0]
            if not sequence.tokens or len(sequence.tokens) != len(sequence.logprobs) or any(
                value is None or not math.isfinite(value) for value in sequence.logprobs
            ):
                raise ValueError("SFT sampled token IDs and logprobs did not align")
            self.first_prefix = list(tokens)
            self.generated_tokens = list(sequence.tokens)
            self.sampling_logprobs = list(sequence.logprobs)
            return result

    stage("opening_connections")
    try:
        service = importlib.import_module("02_sft").service(request_timeout_seconds=120)
        policy_sampler = service.create_sampling_client(base_model=model, model_path=weights)
        reference = service.create_sampling_client(base_model=model)
        tokenizer = policy_sampler.get_tokenizer()
        policy = ProbePolicy(policy_sampler, tokenizer, directory, budget_yuan=1)
        policy.task_id = task_id
        local_environment = importlib.import_module("01_trajectory").local_environment
        with local_environment(args.source_root, args.port, directory) as url:
            environment = TraceProbeEnv(base_url=url)
            task = {"task_id": task_id, "prompt": [{"role": "system", "content": SYSTEM_PROMPT}]}
            rollout = collect_for_task(
                task, policy, env_factory=lambda **kwargs: environment, base_url=url,
                max_steps=1, tools=SHOP_TOOL_SCHEMAS,
            )
        if rollout["error"] or rollout["release_error"] or len(rollout["steps"]) != 1:
            error = rollout["error"] or rollout["release_error"] or {}
            raise RuntimeError(
                f"One-turn rollout failed ({rollout['status']}, {error.get('type')}): "
                f"{error.get('message', 'no executed tool action')}"
            )
        stage(
            "sampling_and_environment_verified",
            environment_version=rollout["initial_result"]["environment_version"],
            tool=rollout["steps"][0]["tool_name"],
            reward=rollout["steps"][0]["reward"],
            policy_input_tokens=sum(call["input_tokens_sent"] for call in policy.calls),
            policy_output_tokens=sum(call["output_tokens"] for call in policy.calls),
        )
        after_text = tokenizer.apply_chat_template(
            normalized_messages(rollout["messages"]), tools=SHOP_TOOL_SCHEMAS,
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        after = list(tokenizer.encode(after_text, add_special_tokens=False))
        target = list(tokenizer.encode(environment.target, add_special_tokens=False))
        if not target:
            raise ValueError("TRACE target encoding is empty")
        before_score = score_target(reference, policy.first_prefix, target)
        stage("first_reference_score_verified", before=before_score)
        after_score = score_target(reference, after, target)
        credit, = turn_rewards(
            [before_score["mean_logprob"], after_score["mean_logprob"]],
            outcome_advantage=0, epsilon=0.1, horizon=3, discount=0.8,
            terminal_weight=2,
        )
        stage("reference_scoring_verified", after=after_score, first_turn_credit=credit)
        if args.check_training or args.check_backward:
            stage("restoring_training_checkpoint")
            training = service.create_training_client_from_state(
                checkpoint["best_development_state_path"]
            )
            all_tokens = [*policy.first_prefix, *policy.generated_tokens]
            prefix_labels = len(policy.first_prefix) - 1
            datum = trio.Datum(
                model_input=trio.ModelInput.from_ints(all_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": np.asarray(all_tokens[1:], dtype=np.int64),
                    "logprobs": np.asarray(
                        [0.0] * prefix_labels + policy.sampling_logprobs, dtype=np.float32
                    ),
                    "advantages": np.asarray(
                        [0.0] * prefix_labels + [0.2 * credit] * len(policy.generated_tokens),
                        dtype=np.float32,
                    ),
                },
            )
            stage("training_checkpoint_restored")
            forward = training.forward([datum], loss_fn="ppo").result()
            metrics = {key: float(value) for key, value in forward.metrics.items()
                       if isinstance(value, (float, int)) and math.isfinite(value)}
            train_logprobs = forward.loss_fn_outputs[0]["logprobs"].data
            if len(train_logprobs) != len(all_tokens) - 1 or any(
                not math.isfinite(value) for value in train_logprobs[prefix_labels:]
            ):
                raise ValueError("PPO returned invalid action logprobs")
            stage("training_forward_verified", training_input_tokens=len(all_tokens) - 1,
                  training_action_tokens=len(policy.generated_tokens), training_metrics=metrics)
            if args.check_backward:
                stage("waiting_for_backward")
                started = time.monotonic()
                backward = training.forward_backward([datum], loss_fn="ppo").result(timeout=150)
                backward_loss = float(backward.metrics["loss_sum"])
                if not math.isfinite(backward_loss):
                    raise ValueError("PPO backward returned a non-finite loss")
                stage("training_backward_verified", backward_loss=backward_loss,
                      backward_seconds=time.monotonic() - started)
        stage("connectivity_verified")
    except Exception as error:
        report.update(failed_stage=report["status"], error_type=type(error).__name__,
                      error_code=getattr(error, "code", None))
        stage("failed")
        raise


if __name__ == "__main__":
    main()
