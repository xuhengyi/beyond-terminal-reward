import importlib
import unittest

metrics = importlib.import_module("05_sync_evaluation").task_metrics


class SyncTests(unittest.TestCase):
    def test_backfill_excludes_future_and_unfinished_task_tokens(self):
        r = {"task_ids": [1,2,3], "results": [
            {"task_id":1,"seconds":2,"reward":1,"steps":2},
            {"task_id":2,"seconds":3,"reward":0,"steps":3}],
             "summary":{"strict_success_task_ids":[1]}}
        calls = [{"task_id":i,"input_tokens_sent":100*i,"output_tokens":i,"cost_no_cache_yuan":.01*i}
                 for i in [1,2,3]]
        first = metrics(r, calls, 1)
        self.assertEqual(first["eval/input_tokens"], 100)
        self.assertEqual(first["eval/output_tokens"], 1)
        self.assertAlmostEqual(first["eval/cost_no_cache_estimated_yuan"], .01)
        self.assertAlmostEqual(first["eval/strict_success_rate_fixed_denominator"], 1/3)
        second = metrics(r, calls, 2)
        self.assertEqual(second["eval/input_tokens"], 300)
        self.assertEqual(second["eval/success_rate_completed"], .5)
        self.assertEqual(second["task/strict_success"], 0)


if __name__ == "__main__":
    unittest.main()
