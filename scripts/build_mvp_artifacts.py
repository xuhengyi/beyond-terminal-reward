#!/usr/bin/env python3
"""Build and validate the fixed, GPU-free inputs for the shopping MVP.

The script deliberately does not load a tokenizer or model. It copies the
selected source records without modifying their prompts, observations, or
labels, then records every selection and integrity check in a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 42
COUNTS = {
    "sft_train": 64,
    "sft_validation": 16,
    "grpo_train": 16,
    "evaluation": 30,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("task_id"), int):
            raise ValueError(f"{path}:{line_number} has no integer task_id")
        rows.append(row)
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate task_id values")
    return rows


def normalized_request(row: dict[str, Any]) -> str | None:
    """Return the SFT user request for within-SFT duplicate detection."""
    for message in row.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return " ".join(message["content"].casefold().split())
    return None


def is_complete_sft_trajectory(row: dict[str, Any]) -> bool:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return False
    actions = [
        call.get("function", {}).get("name")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        if isinstance(call, dict)
    ]
    return bool(actions) and actions[-1] in {"buy_now", "finish_without_purchase"}


def choose(rows: list[dict[str, Any]], count: int, namespace: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"{namespace} has only {len(rows)} eligible rows; need {count}")
    ordered = sorted(rows, key=lambda row: row["task_id"])
    rng = random.Random(f"beyond-terminal-reward:{SEED}:{namespace}")
    return sorted(rng.sample(ordered, count), key=lambda row: row["task_id"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_grpo_parquet(source: Path, selected_task_ids: list[int], output: Path) -> int:
    """Copy selected rows from the canonical veRL parquet without reshaping it."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to create the GRPO parquet; run this script with the pinned upstream .venv"
        ) from exc
    source_table = pq.read_table(source)
    rows_by_id = {row["extra_info"]["task_id"]: row for row in source_table.to_pylist()}
    missing = [task_id for task_id in selected_task_ids if task_id not in rows_by_id]
    if missing:
        raise ValueError(f"GRPO JSONL task IDs missing from canonical parquet: {missing}")
    selected_rows = [rows_by_id[task_id] for task_id in selected_task_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected_rows, schema=source_table.schema), output)
    return source_table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("vendor/shopping-grpo-longhorizon"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mvp"))
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    output = args.output.resolve()

    source_paths = {
        "sft_train": upstream / "data/sft/train.jsonl",
        "sft_validation": upstream / "data/sft/validation.jsonl",
        "grpo_train": upstream / "data/grpo/train.jsonl",
        "evaluation": upstream / "data/evaluation/tasks.jsonl",
    }
    grpo_parquet_source = upstream / "data/grpo/train.parquet"
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not grpo_parquet_source.is_file():
        raise FileNotFoundError(grpo_parquet_source)

    sources = {name: read_jsonl(path) for name, path in source_paths.items()}
    complete_sft = {
        name: [row for row in sources[name] if is_complete_sft_trajectory(row)]
        for name in ("sft_train", "sft_validation")
    }
    selections = {
        "sft_train": choose(complete_sft["sft_train"], COUNTS["sft_train"], "sft_train"),
        "sft_validation": choose(complete_sft["sft_validation"], COUNTS["sft_validation"], "sft_validation"),
        "grpo_train": choose(sources["grpo_train"], COUNTS["grpo_train"], "grpo_train"),
        "evaluation": choose(sources["evaluation"], COUNTS["evaluation"], "evaluation"),
    }

    task_sets = {name: {row["task_id"] for row in rows} for name, rows in selections.items()}
    overlaps = {
        f"{left}__{right}": sorted(task_sets[left] & task_sets[right])
        for index, left in enumerate(task_sets)
        for right in list(task_sets)[index + 1 :]
    }
    if any(overlap for overlap in overlaps.values()):
        raise ValueError(f"MVP split overlap: {overlaps}")

    requests = {}
    for name in ("sft_train", "sft_validation"):
        for row in selections[name]:
            request = normalized_request(row)
            if request is None:
                raise ValueError(f"{name} task {row['task_id']} has no user request")
            requests.setdefault(request, []).append((name, row["task_id"]))
    request_duplicates = {
        request: entries for request, entries in requests.items() if len(entries) > 1
    }
    if request_duplicates:
        raise ValueError(f"MVP SFT request duplicates: {request_duplicates}")

    data_dir = output / "data"
    output_paths = {
        "sft_train": data_dir / "sft-train.jsonl",
        "sft_validation": data_dir / "sft-validation.jsonl",
        "grpo_train": data_dir / "grpo-train.jsonl",
        "evaluation": data_dir / "evaluation.jsonl",
    }
    for name, path in output_paths.items():
        write_jsonl(path, selections[name])
    grpo_parquet_path = data_dir / "grpo-train.parquet"
    grpo_parquet_source_rows = write_grpo_parquet(
        grpo_parquet_source,
        [row["task_id"] for row in selections["grpo_train"]],
        grpo_parquet_path,
    )

    upstream_head = (upstream / ".git").is_dir()
    manifest = {
        "schema_version": "beyond-terminal-reward-mvp-v1",
        "seed": SEED,
        "source_root": str(upstream),
        "source_commit": None,
        "selection_policy": {
            "sft": "complete trajectories ending in buy_now or finish_without_purchase; deterministic task_id sampling",
            "grpo_and_evaluation": "deterministic task_id sampling without model execution",
            "request_duplicate_check": "normalized user requests are unique across the selected SFT train and validation rows",
        },
        "source_files": {
            name: {"path": str(path), "rows": len(sources[name]), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "eligible_sft_rows": {name: len(rows) for name, rows in complete_sft.items()},
        "splits": {
            name: {
                "path": str(path.relative_to(output)),
                "rows": len(selections[name]),
                "task_ids": [row["task_id"] for row in selections[name]],
                "sha256": sha256(path),
            }
            for name, path in output_paths.items()
        },
        "integrity": {
            "task_id_overlaps": overlaps,
            "sft_normalized_request_duplicates": request_duplicates,
            "sft_terminal_action_counts": {
                name: Counter(
                    next(
                        call.get("function", {}).get("name")
                        for message in reversed(row["messages"])
                        if message.get("role") == "assistant" and message.get("tool_calls")
                        for call in message["tool_calls"]
                    )
                    for row in selections[name]
                )
                for name in ("sft_train", "sft_validation")
            },
        },
    }
    manifest["source_files"]["grpo_train_parquet"] = {
        "path": str(grpo_parquet_source),
        "rows": grpo_parquet_source_rows,
        "sha256": sha256(grpo_parquet_source),
    }
    manifest["splits"]["grpo_train_parquet"] = {
        "path": str(grpo_parquet_path.relative_to(output)),
        "rows": len(selections["grpo_train"]),
        "task_ids": [row["task_id"] for row in selections["grpo_train"]],
        "sha256": sha256(grpo_parquet_path),
    }
    if upstream_head:
        import subprocess

        manifest["source_commit"] = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
