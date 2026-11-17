import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

module = importlib.import_module("04_evaluate_pilot")


class Tokenizer:
    def __init__(self, count=100):
        self.count = count
    def apply_chat_template(self, messages, **kwargs):
        return str(messages)
    def encode(self, text, **kwargs):
        return list(range(self.count))


class Sampler:
    def __init__(self):
        self.requests = 0
    def sample(self, **kwargs):
        self.requests += 1
        seq = SimpleNamespace(text='<tool_call>{"name":"search_products","arguments":{"query":"杯子"}}</tool_call>',
                              tokens=list(range(20)), stop_reason="stop")
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[seq], input_tokens=100, output_tokens=20))


class PilotTests(unittest.TestCase):
    def test_499_retries_are_bounded_and_record_unknown_billing(self):
        from pytrio.errors.actor import ActorAPIError
        with TemporaryDirectory() as d:
            sampler=Sampler()
            sampler.sample=lambda **kwargs: (_ for _ in ()).throw(ActorAPIError('HTTP499',status_code=499))
            p=module.SamplingPolicy(sampler,Tokenizer(),Path(d))
            with patch.object(module.time,'sleep') as sleep:
                with self.assertRaises(ActorAPIError):
                    p.complete([],[])
            self.assertEqual(len(p.request_failures),3)
            self.assertEqual(sleep.call_count,2)
            self.assertEqual(p.calls,[])

    def test_non499_is_not_retried(self):
        from pytrio.errors.actor import ActorAPIError
        with TemporaryDirectory() as d:
            sampler=Sampler()
            sampler.sample=lambda **kwargs: (_ for _ in ()).throw(ActorAPIError('HTTP403',status_code=403))
            p=module.SamplingPolicy(sampler,Tokenizer(),Path(d))
            with patch.object(module.time,'sleep') as sleep:
                with self.assertRaises(ActorAPIError):
                    p.complete([],[])
            self.assertEqual(sleep.call_count,0)
            self.assertEqual(p.request_failures,[])

    def test_499_recovery_records_one_returned_response(self):
        from pytrio.errors.actor import ActorAPIError
        with TemporaryDirectory() as d:
            sampler=Sampler()
            original=sampler.sample
            count=[0]
            def sample(**kwargs):
                count[0]+=1
                if count[0]==1:raise ActorAPIError('HTTP499',status_code=499)
                return original(**kwargs)
            sampler.sample=sample
            p=module.SamplingPolicy(sampler,Tokenizer(),Path(d))
            tools = [{"function":{"name":"search_products","parameters":{"required":["query"],"properties":{"query":{"type":"string"}}}}}]
            with patch.object(module.time,'sleep'):
                p.complete([],tools)
            self.assertEqual(count[0],2)
            self.assertEqual(len(p.calls),1)
            self.assertEqual(len(p.request_failures),1)

    def replay_fixture(self):
        response = {"role":"assistant","content":"", "tool_calls":[{"id":"old1","type":"function","function":{"name":"search_products","arguments":"{\"query\":\"杯子\"}"}}]}
        history = [{"role":"user","content":"买杯子"}, response,
                   {"role":"tool","tool_call_id":"old1","content":"原始搜索结果"}]
        return response, history

    def test_resume_replays_without_model_call_and_continues_once(self):
        response, history = self.replay_fixture()
        with TemporaryDirectory() as d:
            sampler = Sampler()
            p = module.SamplingPolicy(sampler, Tokenizer(), Path(d))
            p.task_id = 7
            p.calls = [{"task_id":7,"input_tokens_sent":50,"cost_no_cache_yuan":.5}]
            p.start_replay({"task_id":7,"messages":history})
            self.assertEqual(p.complete(history[:1], []), response)
            self.assertEqual(sampler.requests, 0)
            self.assertEqual(len(p.calls), 1)
            tools = [{"function":{"name":"search_products","parameters":{"required":["query"],"properties":{"query":{"type":"string"}}}}}]
            p.complete(history, tools)
            self.assertEqual(sampler.requests, 1)
            self.assertEqual(len(p.calls), 2)
            self.assertAlmostEqual(p.calls[0]['cost_no_cache_yuan'], .5)

    def test_resume_observation_mismatch_stops_before_paid_request(self):
        response, history = self.replay_fixture()
        with TemporaryDirectory() as d:
            sampler = Sampler()
            p = module.SamplingPolicy(sampler, Tokenizer(), Path(d))
            p.task_id = 7
            p.calls = [{"task_id":7,"input_tokens_sent":50,"cost_no_cache_yuan":.5}]
            p.start_replay({"task_id":7,"messages":history})
            p.complete(history[:1], [])
            changed = history[:-1] + [{**history[-1],"content":"不同的搜索结果"}]
            with self.assertRaises(module.ReplayMismatchError):
                p.complete(changed, [])
            self.assertEqual(sampler.requests, 0)
            self.assertEqual(len(p.calls), 1)

    def test_context_limit_prevents_paid_request(self):
        with TemporaryDirectory() as d:
            sampler = Sampler()
            p = module.SamplingPolicy(sampler, Tokenizer(32768), Path(d))
            with self.assertRaisesRegex(ValueError, "完整上下文"):
                p.complete([], [])
            self.assertEqual(sampler.requests, 0)

    def test_budget_limit_prevents_paid_request(self):
        with TemporaryDirectory() as d:
            sampler = Sampler()
            p = module.SamplingPolicy(sampler, Tokenizer(), Path(d), budget_yuan=0)
            with self.assertRaises(module.PilotBudgetExceeded):
                p.complete([], [])
            self.assertEqual(sampler.requests, 0)

    def test_records_exact_sent_and_generated_tokens(self):
        with TemporaryDirectory() as d:
            p = module.SamplingPolicy(Sampler(), Tokenizer(), Path(d))
            p.task_id = 123
            tools = [{"function":{"name":"search_products","parameters":{"required":["query"],"properties":{"query":{"type":"string"}}}}}]
            assistant = p.complete([{"role":"user","content":"买杯子"}], tools)
            self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], '{"query": "杯子"}')
            self.assertEqual(p.calls[0]["input_tokens_sent"], 100)
            self.assertEqual(p.calls[0]["output_tokens"], 20)
            self.assertAlmostEqual(p.calls[0]["cost_no_cache_yuan"], (100*1.49+20*4.54)/1e6)


if __name__ == "__main__":
    unittest.main()
