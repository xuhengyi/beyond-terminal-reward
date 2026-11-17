#!/usr/bin/env python3
"""Validate MVP data artifacts without importing model or GPU dependencies."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def read_task_ids(path: Path) -> list[int]:
    return [json.loads(line)["task_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/mvp"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_counts = {"sft_train": 64, "sft_validation": 16, "grpo_train": 16, "evaluation": 30}
    task_sets: dict[str, set[int]] = {}
    for name, expected in expected_counts.items():
        split = manifest["splits"][name]
        path = root / split["path"]
        actual_ids = read_task_ids(path)
        if len(actual_ids) != expected or len(set(actual_ids)) != expected:
            raise SystemExit(f"{name}: expected {expected} unique rows, found {len(actual_ids)}")
        if actual_ids != split["task_ids"]:
            raise SystemExit(f"{name}: task IDs differ from manifest")
        if sha256(path) != split["sha256"]:
            raise SystemExit(f"{name}: hash differs from manifest")
        task_sets[name] = set(actual_ids)
    for left, left_ids in task_sets.items():
        for right, right_ids in task_sets.items():
            if left < right and left_ids & right_ids:
                raise SystemExit(f"overlap between {left} and {right}")
    if any(manifest["integrity"]["task_id_overlaps"].values()):
        raise SystemExit("manifest recorded an overlap")
    if manifest["integrity"]["sft_normalized_request_duplicates"]:
        raise SystemExit("manifest recorded an SFT request duplicate")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to validate the GRPO parquet") from exc
    parquet_split = manifest["splits"]["grpo_train_parquet"]
    parquet_path = root / parquet_split["path"]
    parquet_ids = [row["extra_info"]["task_id"] for row in pq.read_table(parquet_path).to_pylist()]
    if parquet_ids != manifest["splits"]["grpo_train"]["task_ids"]:
        raise SystemExit("GRPO parquet task IDs differ from the frozen JSONL selection")
    if sha256(parquet_path) != parquet_split["sha256"]:
        raise SystemExit("GRPO parquet hash differs from manifest")
    print("MVP data artifacts valid: 64/16/16/30, unique task IDs, hashes match.")


if __name__ == "__main__":
    main()
