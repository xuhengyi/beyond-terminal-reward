"""SwanLab 只消费真实指标；连接检查不会创建 PyTRIO 训练任务。"""

from datetime import datetime, timezone
from pathlib import Path

from common import write_json


class Tracker:
    def __init__(self, args, directory: Path, config: dict):
        self.run = None
        self.url = None
        self.mode = args.swanlab_mode
        self.run_id = None
        if self.mode == "disabled":
            return
        import swanlab
        self.run = swanlab.init(
            project=args.swanlab_project,
            name=args.run_name or directory.name,
            mode=self.mode,
            on_online_error="raise",
            public=False,
            log_dir=str(directory / "swanlog"),
            config=config,
            settings=swanlab.Settings(interactive=False),
        )
        self.mode = self.run.mode
        self.run_id = self.run.id
        if self.mode == "online":
            self.url = self.run.url
        write_json(directory / "swanlab.json", self.metadata())

    def metadata(self):
        return {"mode": self.mode, "run_id": self.run_id, "url": self.url}

    def log(self, values: dict, step: int):
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self, error=None):
        if self.run is not None:
            self.run.finish(
                state="crashed" if error is not None else "success",
                error=type(error).__name__ if error is not None else None,
            )


def check_connection(args):
    directory = args.output_dir / "logging_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    if args.swanlab_mode == "disabled":
        raise ValueError("连接检查请指定 --swanlab-mode online 或 local")
    tracker = Tracker(args, directory, {"purpose": "logging_connection_check", "training_executed": False})
    try:
        tracker.log({"connection/check": 1, "connection/training_executed": 0}, step=0)
        result = {**tracker.metadata(), "training_executed": False,
                  "metric": "connection/check", "metric_value": 1}
        write_json(directory / "result.json", result)
        print(result)
    except BaseException as error:
        tracker.finish(error)
        raise
    else:
        tracker.finish()
