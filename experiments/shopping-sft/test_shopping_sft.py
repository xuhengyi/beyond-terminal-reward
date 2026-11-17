"""训练前需要守住的语义：监督边界、右移、轨迹权重、工具参数和上下文长度。"""

import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import EncodedTrajectory, encode_trajectory, normalized_messages, shifted_fields


class CharacterTokenizer:
    """用可读字符级模板建立独立的监督位置预期，不依赖下载或远程服务。"""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt):
        text = ""
        for message in messages:
            body = message.get("content") or ""
            if message.get("tool_calls"):
                body += json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
            text += f"<{message['role']}>" + body + "</end>"
        return text + ("<assistant>" if add_generation_prompt else "")


def fixture():
    return {"task_id": 7, "tools": [], "messages": [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "请买商品"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "buy_now", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "商品观察"},
    ]}


class ShoppingSFTTests(unittest.TestCase):
    def test_empty_assistant_body_still_supervises_tool_call_and_excludes_context(self):
        tokenizer = CharacterTokenizer()
        row = fixture()
        example = encode_trajectory(row, tokenizer, 10000)
        messages = normalized_messages(row["messages"])
        expected = json.dumps(messages[2]["tool_calls"], ensure_ascii=False, sort_keys=True) + "</end>"
        supervised = "".join(chr(token) for token, active in zip(example.tokens, example.mask) if active)
        self.assertEqual(supervised, expected)
        self.assertNotIn("商品观察", supervised)
        self.assertNotIn("系统提示", supervised)

    def test_target_and_mask_shift_together(self):
        example = EncodedTrajectory(1, [10, 11, 20, 21, 30], [0, 0, 1, 1, 0], [])
        fields = shifted_fields(example)
        self.assertEqual(fields["input_tokens"], [10, 11, 20, 21])
        self.assertEqual(fields["target_tokens"], [11, 20, 21, 30])
        self.assertEqual(fields["weights"], [0, 0.5, 0.5, 0])

    def test_long_and_short_trajectories_have_equal_total_contribution(self):
        short = EncodedTrajectory(1, [0, 1, 1], [0, 1, 1], [])
        long = EncodedTrajectory(2, [0] + [1] * 200, [0] + [1] * 200, [])
        short_weights = shifted_fields(short, batch_size=2)["weights"]
        long_weights = shifted_fields(long, batch_size=2)["weights"]
        self.assertAlmostEqual(sum(short_weights), 0.5)
        self.assertAlmostEqual(sum(long_weights), 0.5)
        # 两条轨迹动作 NLL 分别为 2 和 4，按轨迹平均应为 3，不能被长轨迹主导。
        self.assertAlmostEqual(sum(short_weights) * 2 + sum(long_weights) * 4, 3)

    def test_overlong_trajectory_is_rejected_instead_of_losing_terminal_action(self):
        with self.assertRaisesRegex(ValueError, "拒绝截断"):
            encode_trajectory(fixture(), CharacterTokenizer(), max_length=10)

    def test_unmatched_tool_result_is_rejected(self):
        row = fixture()
        row["messages"][-1]["tool_call_id"] = "wrong_call"
        with self.assertRaisesRegex(ValueError, "tool_call_id"):
            encode_trajectory(row, CharacterTokenizer(), max_length=10000)

    def test_gradient_is_zero_for_observations(self):
        import torch
        example = EncodedTrajectory(1, [10, 11, 20, 21, 30], [0, 0, 1, 1, 0], [])
        weights = torch.tensor(shifted_fields(example)["weights"])
        logprobs = torch.tensor([-1., -2., -3., -4.], requires_grad=True)
        loss = -(weights * logprobs).sum()
        loss.backward()
        self.assertEqual(logprobs.grad.tolist(), [0., -0.5, -0.5, 0.])

    def test_qwen_tool_parser_checks_schema_and_single_call(self):
        parse_call = importlib.import_module("03_check_action").parse_call
        tools = [{"function": {"name": "search_products", "parameters": {"required": ["query"], "properties": {"query": {"type": "string"}}}}}]
        text = "<tool_call><function=search_products><parameter=query>金属模具</parameter></function></tool_call>"
        self.assertEqual(parse_call(text, tools), ("search_products", {"query": "金属模具"}))
        with self.assertRaises(ValueError):
            parse_call(text + text, tools)
        with self.assertRaises(ValueError):
            parse_call(text.replace("parameter=query", "parameter=unknown"), tools)

    def test_insufficient_balance_records_no_training_without_retry(self):
        from pytrio.errors.control import ControlAPIError
        module = importlib.import_module("02_sft")
        class UnfundedClient:
            creations = 0
            def get_server_capabilities(self):
                return SimpleNamespace(supported_models=[SimpleNamespace(
                    model_name="test-model", max_context_length=10000,
                    training=SimpleNamespace(available=True, max_seq_len=10000),
                )])
            def create_lora_training_client(self, **kwargs):
                self.creations += 1
                raise ControlAPIError("billing insufficient balance", code="billing_insufficient_balance", status_code=409)
        client = UnfundedClient()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = module.parse_args(["train", "--base-model", "test-model", "--max-samples", "1",
                                      "--batch-size", "1", "--output-dir", str(output)])
            paths_and_rows = (Path(__file__), Path(__file__), [fixture()], [{**fixture(), "task_id": 8}])
            with patch.object(module, "service", return_value=client), patch.object(module, "load_splits", return_value=paths_and_rows):
                with self.assertRaisesRegex(ValueError, "未更新参数"):
                    module.train(args)
            files = list((output / "runs").glob("*/result.json"))
            self.assertEqual(len(files), 1)
            result = json.loads(files[0].read_text())
            self.assertEqual(result["error"]["code"], "billing_insufficient_balance")
            self.assertFalse(result["training_executed"])
            self.assertEqual(result["steps"], [])
            self.assertEqual(client.creations, 1)


if __name__ == "__main__":
    unittest.main()
