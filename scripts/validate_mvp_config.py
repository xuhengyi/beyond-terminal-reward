#!/usr/bin/env python3
"""Check the MVP run contract against the pinned upstream GRPO defaults."""

from __future__ import annotations

import json
import re
from pathlib import Path


def yaml_scalar(source: str, dotted_key: str) -> str:
    """Extract one scalar from the small pinned YAML contract without PyYAML."""
    parts = dotted_key.split(".")
    level = 0
    for raw_line in source.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        match = re.match(r"\s*([A-Za-z_][\w-]*):\s*(.*?)\s*$", raw_line)
        if not match:
            continue
        key, value = match.groups()
        current_level = indent // 2
        if current_level == level and key == parts[level]:
            if level == len(parts) - 1:
                return value
            level += 1
            continue
        if current_level <= level:
            level = current_level
    raise KeyError(dotted_key)


def main() -> None:
    root = Path(".").resolve()
    run = json.loads((root / "configs/mvp-run.json").read_text(encoding="utf-8"))
    upstream = root / run["upstream"]["path"]
    actual_commit = (
        __import__("subprocess").check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    )
    if actual_commit != run["upstream"]["commit"]:
        raise SystemExit(f"upstream commit mismatch: {actual_commit} != {run['upstream']['commit']}")
    if run["model"]["revision"].startswith("REPLACE_"):
        print("WARN: Hugging Face model revision is not locked yet; lock it before downloading weights.")
    if run["grpo_common"]["advantage_standard_deviation_normalization"]:
        raise SystemExit("MVP requires advantage standard-deviation normalization disabled")
    if run["grpo_common"]["length_shaping"]:
        raise SystemExit("MVP requires length shaping disabled")
    if run["trace"]["enabled_for_terminal_branch"]:
        raise SystemExit("Terminal branch must keep TRACE disabled")
    if not run["trace"]["enabled_for_trace_branch"]:
        raise SystemExit("TRACE branch must enable TRACE")
    if run["grpo_common"]["rollouts_per_prompt"] != 4:
        raise SystemExit("MVP needs four rollouts per prompt")
    if run["evaluation"]["methods"] != ["base", "sft", "terminal_grpo", "trace_grpo"]:
        raise SystemExit("MVP evaluation methods differ from the four-group plan")

    source = (upstream / "configs/grpo.yaml").read_text(encoding="utf-8")
    expected_upstream = {
        "algorithm.norm_adv_by_std_in_grpo": "false",
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.rollout.n": "4",
        "actor_rollout_ref.rollout.temperature": "0.7",
        "actor_rollout_ref.rollout.top_p": "0.9",
    }
    for key, expected in expected_upstream.items():
        actual = yaml_scalar(source, key)
        if actual != expected:
            raise SystemExit(f"pinned upstream {key}={actual!r}; expected {expected!r}")
    print("MVP configuration valid: pinned commit and Terminal/TRACE branch controls match the plan.")


if __name__ == "__main__":
    main()
