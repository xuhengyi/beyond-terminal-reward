"""验证训练真实聚合、跨轮覆盖、开发集隔离和连接失败边界。"""

import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import EncodedTrajectory
from training_loop import epoch_batches, fit, learning_rate_at, weighted_nll


def example(task_id, length=4):
    return EncodedTrajectory(task_id, [0] + [1] * length, [0] + [1] * length, [])


def result_for(examples, value=2):
    return SimpleNamespace(loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-value] * (len(x.tokens)-1))}
                                          for x in examples], metrics={})


class Future:
    def __init__(self, value):
        self.value = value
    def result(self):
        return self.value


class FakeTraining:
    def __init__(self):
        self.backward_ids = []
        self.forward_ids = []
    def forward_backward(self, rows, **kwargs):
        self.backward_ids.extend(x.task_id for x in rows)
        return Future(result_for(rows))
    def forward(self, rows, **kwargs):
        self.forward_ids.extend(x.task_id for x in rows)
        return Future(result_for(rows, 3))
    def optim_step(self, adam):
        return Future(None)
    def save_state(self, name):
        return Future(SimpleNamespace(path="state:" + name))
    def save_weights_for_sampler(self, name):
        return Future(SimpleNamespace(path="sampler:" + name))


class TrainingTests(unittest.TestCase):
    def test_nll_uses_actual_logprobs_and_trajectory_weights(self):
        rows = [example(1, 2), example(2, 100)]
        result = result_for(rows)
        result.loss_fn_outputs[1]["logprobs"].data = [-4.] * 100
        self.assertAlmostEqual(weighted_nll(result, rows), 3.)
        result.loss_fn_outputs[0]["logprobs"].data = [float("nan")] * 2
        with self.assertRaises(ValueError):
            weighted_nll(result, rows)

    def test_all_rows_are_seen_once_per_epoch(self):
        rows = [example(n) for n in range(5)]
        batches = list(epoch_batches(rows, 2, 3, 42))
        for epoch in [1, 2, 3]:
            ids = [x.task_id for e, batch, _ in batches if e == epoch for x in batch]
            self.assertCountEqual(ids, range(5))
            self.assertEqual(sum(last for e, _, last in batches if e == epoch), 1)

    def test_linear_schedule_matches_zero_based_trainer_warmup(self):
        self.assertEqual(learning_rate_at(1, 300, 1e-4, "linear", .03), 1e-12)
        self.assertAlmostEqual(learning_rate_at(10, 300, 1e-4, "linear", .03), 1e-4)
        self.assertGreater(learning_rate_at(300, 300, 1e-4, "linear", .03), 0)
        self.assertLess(learning_rate_at(300, 300, 1e-4, "linear", .03), 1e-6)

    def test_validation_never_enters_backward_and_epochs_are_saved(self):
        self.check_fit(6, True)

    def test_step_limit_is_reported_as_partial_epochs(self):
        self.check_fit(3, False)

    def check_fit(self, steps, completed):
        training = FakeTraining()
        report = {"steps": []}
        args = SimpleNamespace(steps=steps, epochs=3, batch_size=2, seed=42,
                               learning_rate=1e-4, lr_schedule="constant", warmup_ratio=.03,
                               train_price_per_million=4.54)
        with TemporaryDirectory() as tmp, patch("training_loop.to_datum", side_effect=lambda x, n: x):
            fit(training, [example(n) for n in range(4)], [example(100), example(101)],
                args, report, Path(tmp), SimpleNamespace(log=lambda *a, **kw: None))
        self.assertFalse(set(training.backward_ids) & {100, 101})
        self.assertEqual(set(training.forward_ids), {100, 101})
        self.assertEqual(len(report["steps"]), steps)
        self.assertEqual(report["full_epochs_completed"], completed)
        self.assertTrue(all(x["validation_nll"] == 3 for x in report["epochs"]))
        if completed:
            self.assertEqual(len(report["epochs"]), 3)
            for n in range(4):
                self.assertEqual(training.backward_ids.count(n), 3)

    def test_online_logging_failure_cannot_create_training(self):
        module = importlib.import_module("02_sft")
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = module.parse_args(["train", "--output-dir", tmp, "--swanlab-mode", "online"])
            client = SimpleNamespace(
                get_server_capabilities=lambda: SimpleNamespace(supported_models=[SimpleNamespace(
                    model_name=args.base_model, max_context_length=32768,
                    training=SimpleNamespace(available=True, max_seq_len=32768))]),
            )
            paths = (Path(__file__), Path(__file__), [{"task_id": 1}], [{"task_id": 2}])
            with patch.object(module, "service", return_value=client), patch.object(module, "load_splits", return_value=paths), \
                 patch.object(module, "Tracker", side_effect=ValueError("not online")):
                with self.assertRaisesRegex(ValueError, "not online"):
                    module.train(args)
            records = list(directory.glob("runs/*/result.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text())
            self.assertFalse(record["training_executed"])
            self.assertEqual(record["status"], "logging_failed_before_training")

    def test_full_recipe_cannot_be_silently_shortened(self):
        module = importlib.import_module("02_sft")
        args = module.parse_args(["train", "--require-complete-epochs", "--max-samples", "4",
                                  "--epochs", "3", "--batch-size", "2", "--steps", "2"])
        paths = (Path(__file__), Path(__file__), [{"task_id": n} for n in range(4)], [{"task_id": 100}])
        with patch.object(module, "load_splits", return_value=paths), patch.object(module, "service") as service:
            with self.assertRaisesRegex(ValueError, "steps 上限不足"):
                module.train(args)
            service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
