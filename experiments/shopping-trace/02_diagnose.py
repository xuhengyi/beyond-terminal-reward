"""Audit frozen-reference TRACE credit on recorded and controlled development cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import unicodedata

preflight = importlib.import_module("01_preflight")
from common import DEFAULT_SOURCE, normalized_messages, use_source, write_json


def load_cases():
    source = preflight.PROJECT / "experiments/shopping-sft/outputs/evaluation_pilots"
    cases = [json.loads(path.read_text()) for path in sorted(source.glob("*/task_*.json"))]
    by_id = {case["task_id"]: case for case in cases if case["done"]}
    return by_id


def calls(trajectory):
    return [deepcopy(message) for message in trajectory["messages"] if message["role"] == "assistant"]


def action(name, arguments, index):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"audit_{index}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }]}


class ReplayPolicy:
    def __init__(self, messages):
        self.messages = iter(messages)

    def complete(self, messages, tools):
        return next(self.messages)


def normalize_option(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def find_alternative(cases, url):
    from shopping_grpo.environment.client import ShopAgentEnv
    from shopping_grpo.environment.tools import tool_call_to_action

    attempts = 0
    for base in cases.values():
        goal = base["terminal_result"]["goal"]
        search = base["steps"][0]
        if search["tool_name"] != "search_products":
            continue
        options = [normalize_option(value) for value in goal["goal_options"]]
        for product in search["result"]["observation_state"]["products"][:6]:
            asin = product["asin"]
            if asin == goal["asin"] or product.get("category") != goal["category"]:
                continue
            attempts += 1
            sequence = [calls(base)[0], action("open_product", {"asin": asin}, 1)]
            with ShopAgentEnv(base_url=url) as env:
                env.reset(base["task_id"])
                env.step(search["env_action"])
                opened = env.step(tool_call_to_action("open_product", {"asin": asin}))
                state = opened["observation_state"]
                if state.get("product", {}).get("asin") != asin:
                    continue
                available = {
                    normalize_option(value): value
                    for values in state.get("available_options", {}).values()
                    for value in values
                }
                if any(option not in available for option in options):
                    continue
                for index, option in enumerate(options, start=2):
                    args = {"value": available[option]}
                    sequence.append(action("select_option", args, index))
                    env.step(tool_call_to_action("select_option", args))
                    if env.done:
                        break
                if env.done:
                    continue
                end = env.step(tool_call_to_action("buy_now", {}))
                if end.get("termination_reason") == "valid_alternative_purchase":
                    sequence.append(action("buy_now", {}, len(sequence)))
                    return (base, sequence), attempts
    return None, attempts


def replay(base, sequence, url):
    from shopping_grpo.evaluation.rollout import collect_for_task

    task = {"task_id": base["task_id"], "prompt": base["messages"][:1]}
    result = collect_for_task(task, ReplayPolicy(sequence), base_url=url, max_steps=len(sequence))
    if result["error"] or result["release_error"] or result["blocked_tool_calls"] or not result["done"]:
        error = result["error"] or result["release_error"] or {}
        raise RuntimeError(f"Controlled replay failed: {result['status']} {error.get('message', '')}")
    return result


def prefixes(tokenizer, case, tools):
    messages = case["messages"]
    first = next(i for i, message in enumerate(messages) if message["role"] == "assistant")
    boundaries = [first, *(i + 1 for i, message in enumerate(messages) if message["role"] == "tool")]
    if len(boundaries) != len(case["steps"]) + 1:
        raise ValueError("Unexpected action and tool-response layout")
    for index, boundary in enumerate(boundaries):
        text = tokenizer.apply_chat_template(
            normalized_messages(messages[:boundary]), tools=tools,
            tokenize=False, add_generation_prompt=index == 0, enable_thinking=False,
        )
        yield tuple(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-result", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--alternative-candidates", type=Path)
    parser.add_argument("--port", type=int, default=5703)
    args = parser.parse_args()
    use_source(DEFAULT_SOURCE)
    from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
    from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset
    from shopping_grpo.training.grpo.trace import canonical_purchase_target, turn_rewards

    cases = load_cases()
    _, blind_ids = validate_canonical_blind_asset()
    if set(cases).intersection(blind_ids):
        raise ValueError("TRACE diagnostic includes a held-out task")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(__file__).resolve().parent / "outputs/diagnostic" / run_id
    report_path = directory / "result.json"
    report = {"status": "preparing", "parameter_updates": 0, "cases": []}
    write_json(report_path, report)
    print(report_path, flush=True)

    base = cases[22263]
    correct = calls(base)
    wrong = [*correct[:2], action("select_option", {"value": "m"}, 2), correct[3]]
    repeated = [
        correct[0], action("back_to_search", {}, 1),
        action("search_products", base["steps"][0]["parameters"], 2), *correct[1:],
    ]
    environment = importlib.import_module("01_trajectory").local_environment
    with environment(DEFAULT_SOURCE, args.port, directory) as url:
        cases_to_score = [
            ("gold_with_spec_checks", cases[1038]),
            ("partial_alternative", cases[15097]),
            ("gold_size_l", base),
            ("wrong_size_m", replay(base, wrong, url)),
            ("repeated_search", replay(base, repeated, url)),
        ]
        if args.alternative_candidates:
            teacher_path = DEFAULT_SOURCE / "data/sft/validation.jsonl"
            teachers = {row["task_id"]: row for row in (
                json.loads(line) for line in teacher_path.read_text().splitlines() if line
            )}
            candidates = json.loads(args.alternative_candidates.read_text())
            alternative = None
            attempts = len(candidates)
            for candidate in candidates:
                task_id = candidate["task_id"]
                if task_id in blind_ids or task_id not in teachers:
                    raise ValueError("Catalog alternative is not a development task")
                teacher = teachers[task_id]
                sequence = [
                    action("search_products", {"query": candidate["query"]}, 0),
                    action("open_product", {"asin": candidate["asin"]}, 1),
                ]
                for option in candidate["options"]:
                    sequence.append(action("select_option", {"value": option}, len(sequence)))
                sequence.append(action("buy_now", {}, len(sequence)))
                try:
                    valid = replay(teacher, sequence, url)
                    gold = replay(teacher, calls(teacher), url)
                except RuntimeError as error:
                    print(f"Candidate {task_id}: {error}", flush=True)
                    continue
                if valid["terminal_result"].get("termination_reason") != "valid_alternative_purchase":
                    continue
                if gold["terminal_result"].get("termination_reason") != "gold_purchase":
                    continue
                alternative = True
                cases_to_score.extend([(f"gold_pair_{task_id}", gold),
                                       (f"valid_alternative_{task_id}", valid)])
        else:
            alternative, attempts = find_alternative(cases, url)
            if alternative is None:
                teacher_path = DEFAULT_SOURCE / "data/sft/validation.jsonl"
                teachers = [json.loads(line) for line in teacher_path.read_text().splitlines() if line]
                for teacher in teachers[:20]:
                    if teacher["task_id"] in blind_ids or teacher["task_id"] in cases:
                        continue
                    sequence = calls(teacher)
                    try:
                        verified = replay(teacher, sequence, url)
                    except RuntimeError:
                        continue
                    if verified["terminal_result"].get("termination_reason") != "gold_purchase":
                        continue
                    alternative, checked = find_alternative({teacher["task_id"]: verified}, url)
                    attempts += checked
                    if alternative:
                        break
            if alternative:
                cases_to_score.append(("valid_alternative", replay(*alternative, url)))
    report["local_alternative_candidates_checked"] = attempts
    report["valid_alternative_found"] = alternative is not None
    report["status"] = "scoring"
    if args.alternative_candidates:
        report["alternative_candidates"] = str(args.alternative_candidates)
    write_json(report_path, report)
    print(f"{len(cases_to_score)} cases; valid alternative found: {alternative is not None}", flush=True)

    checkpoint = json.loads(args.run_result.read_text())
    service = importlib.import_module("02_sft").service(request_timeout_seconds=120)
    reference = service.create_sampling_client(base_model=checkpoint["base_model"])
    tokenizer = reference.get_tokenizer()
    report["reference"] = "frozen_base_model"
    cached_scores = {}
    prior = {}
    if args.prior_report:
        prior = {item["case"]: item for item in json.loads(args.prior_report.read_text())["cases"]}
        report["prior_report"] = str(args.prior_report)
    reused = set()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name, case in cases_to_score:
            terminal = case["terminal_result"]
            goal = terminal["goal"]
            target_text = canonical_purchase_target(goal["asin"], goal["goal_options"])
            target = tuple(tokenizer.encode(target_text, add_special_tokens=False))
            states = list(prefixes(tokenizer, case, SHOP_TOOL_SCHEMAS))
            if any(len(state) + len(target) > 24576 for state in states):
                raise ValueError("Diagnostic trajectory exceeds the original TRACE context limit")
            keys = [(state, target) for state in states]
            saved = prior.get(name)
            if saved:
                if saved["task_id"] != case["task_id"] or len(saved["prefix_scores"]) != len(keys):
                    raise ValueError("Saved scores do not match the diagnostic case")
                for key, result in zip(keys, saved["prefix_scores"], strict=True):
                    if result["prefix_tokens"] != len(key[0]) or result["target_tokens"] != len(key[1]):
                        raise ValueError("Saved score lengths do not match the diagnostic prefix")
                    cached_scores[key] = result
                    reused.add(key)
            for key in keys:
                if key not in cached_scores:
                    cached_scores[key] = executor.submit(preflight.score_target, reference, *key)
            results = [item if isinstance(item, dict) else item.result()
                       for item in (cached_scores[key] for key in keys)]
            scores = [result["mean_logprob"] for result in results]
            gaps = [0.1 - score for score in scores]
            deltas = [math.log(left / right) for left, right in zip(gaps[:-1], gaps[1:], strict=True)]
            credit = turn_rewards(scores, outcome_advantage=0, epsilon=0.1, horizon=3,
                                  discount=0.8, terminal_weight=2)
            steps = [
                {"tool": step["tool_name"], "td": delta, "credit_without_outcome": value}
                for step, delta, value in zip(case["steps"], deltas, credit, strict=True)
            ]
            record = {
                "case": name, "task_id": case["task_id"], "reward": case["final_reward"],
                "outcome": terminal["termination_reason"], "prefix_scores": results,
                "cumulative_td": sum(deltas), "steps": steps,
            }
            report["cases"].append(record)
            write_json(report_path, report)
            print(name, record["outcome"], f"reward={record['reward']:.4f}",
                  "TD", [round(delta, 4) for delta in deltas], flush=True)
    report["unique_scored_prefixes"] = len(cached_scores)
    report["reused_scored_prefixes"] = len(reused)
    report["new_scoring_requests"] = len(cached_scores) - len(reused)
    report["scoring_input_tokens"] = sum(
        (item if isinstance(item, dict) else item.result())["scoring_input_tokens"]
        for item in cached_scores.values()
    )
    report["status"] = "completed"
    write_json(report_path, report)


if __name__ == "__main__":
    main()
