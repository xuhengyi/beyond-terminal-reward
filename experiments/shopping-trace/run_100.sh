#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
sft_report="${1:?SFT result required}"
checkpoint="${2:?TRACE checkpoint report required}"
log_dir="${3:?supervisor output directory required}"
mkdir -p -- "$log_dir"
exec >> "$log_dir/console.log" 2>&1
export PYTHONDONTWRITEBYTECODE=1

for attempt in 1 2 3; do
    attempt_log="$log_dir/attempt-$attempt.log"
    echo "TRACE-GRPO launch $attempt from $checkpoint"
    set +e
    .venv/bin/python -u experiments/shopping-trace/03_train_pilot.py \
        --run-result "$sft_report" --updates 100 --resume-from "$checkpoint" 2>&1 | tee "$attempt_log"
    exit_code=${PIPESTATUS[0]}
    set -e
    if (( exit_code == 0 )); then
        echo "TRACE-GRPO completed 100 updates."
        exit 0
    fi

    report="$(sed -n 's/^run report: //p' "$attempt_log")"
    if (( attempt == 3 )) || [[ ! -f "$report" ]]; then
        exit "$exit_code"
    fi
    if ! .venv/bin/python - "$report" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
code = str(report.get("error_code") or "")
recoverable = (
    report["status"] == "failed"
    and report.get("failed_stage") in {"sampling", "scoring", "forward_backward"}
    and (
        report.get("error_type") in {
            "TimeoutError", "ActorAPIError", "TransportError", "ConnectError", "ReadTimeout"
        }
        or (
            report.get("failed_stage") == "forward_backward"
            and code == "training.run_broken"
            and report["steps"]
            and (report.get("pending_update") or {}).get("step") == report["steps"][-1]["step"] + 1
            and (Path(sys.argv[1]).parent / f"step-{report['steps'][-1]['step'] + 1:03d}-datums.json.gz").is_file()
        )
        or (
            report.get("failed_stage") == "scoring"
            and report.get("error_type") == "ControlAPIError"
            and code == "conflict"
            and report.get("error_http_status") == 409
        )
    )
    and not code.startswith(("billing", "auth", "validation"))
    and report["steps"]
    and all(report["steps"][-1].get(key) for key in ("state_path", "sampler_path"))
)
sys.exit(0 if recoverable else 1)
PY
    then
        echo "Stopped at a stage that requires inspection. See $report"
        exit "$exit_code"
    fi
    checkpoint="$report"
    echo "Recovering from the last confirmed optimizer checkpoint."
done
