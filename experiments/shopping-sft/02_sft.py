"""审计完整 assistant 监督数据，运行有边界的 PyTRIO LoRA SFT 与 SwanLab。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from common import (
    DEFAULT_SOURCE, OUTPUTS, encode_trajectory, read_rows, sha256, shifted_fields,
    split_path, to_datum, use_source, write_json,
)
from tracking import Tracker, check_connection
from training_loop import fit


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def prepared_rows(rows: list[dict], tokenizer, max_length: int):
    examples, rejected = [], []
    for row in rows:
        try:
            examples.append(encode_trajectory(row, tokenizer, max_length))
        except ValueError as error:
            rejected.append({"task_id": row["task_id"], "reason": str(error)})
    return examples, rejected


def guard_inputs(args, train_path, dev_path):
    use_source(args.source_root)
    from shopping_grpo.evaluation.blind_guard import guard_blind_final
    guard_blind_final([train_path, dev_path], allowed=False)


def load_splits(args):
    train_path = args.data or split_path(args.source_root, "train")
    dev_path = args.validation_data or split_path(args.source_root, "validation")
    guard_inputs(args, train_path, dev_path)
    train_rows, dev_rows = read_rows(train_path), read_rows(dev_path)
    if {r["task_id"] for r in train_rows} & {r["task_id"] for r in dev_rows}:
        raise ValueError("训练和开发 task_id 有交集")
    return train_path, dev_path, train_rows, dev_rows


def audit(args):
    tokenizer = load_tokenizer(args.tokenizer or args.base_model)
    train_path, dev_path, train_rows, dev_rows = load_splits(args)
    examples, rejected = prepared_rows(train_rows, tokenizer, args.max_length)
    dev_examples, dev_rejected = prepared_rows(dev_rows, tokenizer, args.max_length)
    if not examples or not dev_examples:
        raise ValueError("训练或开发数据没有可用完整轨迹，请检查模板和长度")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_model": args.base_model, "tokenizer": args.tokenizer or args.base_model,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "train_source_sha256": sha256(train_path), "validation_source_sha256": sha256(dev_path),
        "max_length": args.max_length,
        "train": {"source_rows": len(train_rows), "accepted": len(examples), "rejected": rejected},
        "validation": {"source_rows": len(dev_rows), "accepted": len(dev_examples), "rejected": dev_rejected},
        "task_id_overlap": [], "normalization": "mean_of_per_trajectory_action_token_CE",
        "trajectories": [{"task_id": x.task_id, "tokens": len(x.tokens),
                          "action_tokens": x.action_tokens, "assistant_turns": len(x.spans),
                          "weight_sum": sum(shifted_fields(x)["weights"])} for x in examples],
    }
    write_json(args.output_dir / "sft_audit.json", summary)
    # 真正执行 SDK 数据校验，确保准备数据能进入当前 PyTRIO 版本。
    for example in examples:
        to_datum(example)
    example = examples[0]
    fields = shifted_fields(example)
    with (args.output_dir / "token_audit.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["target_position", "input_id", "target_id", "target_text", "supervised", "weight"])
        for position, (input_id, target, weight) in enumerate(zip(
            fields["input_tokens"], fields["target_tokens"], fields["weights"], strict=True,
        ), start=1):
            writer.writerow([position, input_id, target, tokenizer.decode([target]), int(weight > 0), weight])
    lines = ["# 第二课：Action-only LoRA SFT", "",
             f"训练完整轨迹：{len(examples)}/{len(train_rows)}；开发完整轨迹：{len(dev_examples)}/{len(dev_rows)}。", "",
             "## 第一个样本", "",
             f"task_id={example.task_id}；完整长度={len(example.tokens)}；动作 token={example.action_tokens}。", "",
             "| 轮次 | 工具 | 原序列 token 区间（左闭右开） | 监督 token 数 |", "|---|---|---|---|"]
    for turn, span in enumerate(example.spans, start=1):
        lines.append(f"| {turn} | `{span['tool']}` | [{span['start']}, {span['end']}) | {span['end'] - span['start']} |")
    first = example.spans[0]
    lines += ["", "第一轮实际被监督的文本（由目标 tokenizer 解码）：", "", "```text",
              tokenizer.decode(example.tokens[first["start"]:first["end"]]), "```", "",
              "## 为什么右移？", "",
              "input=tokens[:-1]，target=tokens[1:]，weights=mask[1:]/动作token数。",
              "每个输入位置负责预测下一个 token，mask 必须随 target 一起右移。",
              "可在 token_audit.tsv 中检查：工具返回的 target weight 为 0，assistant 工具名和参数 weight 大于 0。", "",
              "## 为什么按轨迹归一化？", "",
              "单条轨迹每个动作 token 的权重为 1/该轨迹动作token数；一条完整轨迹的权重和为 1。",
              "一个 batch 再除以轨迹数，形成每条轨迹先平均、然后对轨迹平均的损失。长示范不会只因为 token 多而占更大权重。", "",
              "## LoRA 与 SFT 分别是什么？", "",
              "SFT 决定学什么：提高已验收示范动作的条件概率。LoRA 决定更新哪些参数：在冻结基座上训练低秩增量。",
              "loss=-mean_轨迹(mean_动作token(log p(token | 历史)))。",
              "环境奖励用于前期验收示范，不会直接乘到本阶段的交叉熵里；那是后续 RL 的学习问题。", "",
              "## 检查自己的理解", "",
              "1. 空正文的 assistant 工具调用，监督 token 数为什么仍大于 0？",
              "2. 工具 observation 的权重为 0，它还能影响后面的动作预测吗？",
              "3. 100 个动作 token 和 1000 个动作 token 的轨迹，各自权重和是多少？",
              "4. 输入位置 k 对应的目标和 mask，应该取原序列的哪个位置？", "",
              "本文件是数据审计，不代表远程训练已完成。真实执行证据保存于 runs/。", ""]
    (args.output_dir / "lesson_02.md").write_text("\n".join(lines))
    print(json.dumps({"train_accepted": len(examples), "train_rejected": len(rejected),
                      "validation_accepted": len(dev_examples), "validation_rejected": len(dev_rejected),
                      "report": str(args.output_dir / "lesson_02.md")}, ensure_ascii=False))


def service(request_timeout_seconds=30):
    import pytrio as trio
    from pytrio.config.config import auth_state
    if not auth_state.api_key:
        raise ValueError("本地尚未登录 PyTRIO。请在项目终端运行 uv run trio login；不要把密钥发到聊天。")
    trio.configure(request_timeout_seconds=request_timeout_seconds, retry_max_attempts=2,
                   actor_capacity_wait_timeout_seconds=120)
    return trio.ServiceClient()


def preflight(args):
    from pytrio.config.config import auth_state
    report = {"credential_present": bool(auth_state.api_key), "models": [], "training_executed": False}
    if auth_state.api_key:
        report["models"] = [m.model_dump() for m in service().get_server_capabilities().supported_models]
    write_json(args.output_dir / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["credential_present"]:
        print("下一步：在项目终端运行 uv run trio login；本次没有创建远程训练任务。")


def action_nll(client, example):
    """固定示范探针的动作平均负对数概率，用于确认更新和重载。"""
    result = client.forward([to_datum(example)], loss_fn="cross_entropy").result()
    logprobs = result.loss_fn_outputs[0]["logprobs"].data
    weights = shifted_fields(example)["weights"]
    if len(logprobs) != len(weights):
        raise ValueError("探针 logprob 没有与预测目标对齐")
    value = -sum(float(lp) * weight for lp, weight in zip(logprobs, weights, strict=True) if weight > 0)
    if not math.isfinite(value):
        raise ValueError("动作探针 NLL 非有限值")
    return value


def train(args):
    import pytrio as trio
    from pytrio.errors.base import PyTrioError

    if min(args.max_samples, args.batch_size, args.steps, args.lora_rank, args.epochs, args.max_validation_samples) <= 0 or args.learning_rate <= 0:
        raise ValueError("样本数、batch、步数、rank 和学习率都必须大于 0")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio 必须在 [0,1) 内")
    path, dev_path, rows, dev_rows = load_splits(args)
    if args.require_complete_epochs and len(rows) != args.max_samples:
        raise ValueError("全量模式要求文件样本数与 max-samples 一致")
    if args.require_complete_epochs and args.steps < args.epochs * math.ceil(len(rows) / args.batch_size):
        raise ValueError("steps 上限不足以完成全部 epochs")
    client = service()
    available = {m.model_name: m for m in client.get_server_capabilities().supported_models}
    if args.base_model not in available or not available[args.base_model].training.available:
        raise ValueError(f"当前账户没有 {args.base_model} 的训练能力；请先运行 preflight 检查模型")
    model = available[args.base_model]
    limits = [x for x in (model.max_context_length, model.training.max_seq_len) if x is not None]
    max_length = min([args.max_length, *limits])
    random.Random(args.seed).shuffle(rows)
    run = args.output_dir / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    report = {"base_model": args.base_model, "seed": args.seed, "lora_rank": args.lora_rank,
              "learning_rate": args.learning_rate, "max_length": max_length,
              "train_source_sha256": sha256(path), "validation_source_sha256": sha256(dev_path),
              "requested_steps": args.steps, "requested_epochs": args.epochs,
              "batch_size": args.batch_size, "lr_schedule": args.lr_schedule, "warmup_ratio": args.warmup_ratio,
              "optimizer": {"beta1": .9, "beta2": .999, "eps": 1e-8, "weight_decay": 0., "grad_clip_norm": 1.},
              "lora_modules": {"train_mlp": True, "train_attn": True, "train_unembed": False},
              "platform_unexposed": ["lora_alpha", "lora_dropout"],
              "steps": [], "rejected": [], "training_executed": False,
              "status": "creating_training_run", "normalization": "mean_of_per_trajectory_action_token_CE"}
    write_json(run / "result.json", report)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        tracker = Tracker(args, run, {**config, **report})
    except BaseException as error:
        report["status"] = "logging_failed_before_training"
        report["failure_type"] = type(error).__name__
        write_json(run / "result.json", report)
        raise
    report["swanlab"] = tracker.metadata()
    try:
        training = client.create_lora_training_client(
            base_model=args.base_model, rank=args.lora_rank, seed=args.seed,
            train_mlp=True, train_attn=True, train_unembed=False,
        )
    except PyTrioError as error:
        report["status"] = "creation_failed"
        report["error"] = {"code": error.code, "http_status": error.status_code}
        write_json(run / "result.json", report)
        tracker.finish(error)
        if error.code == "billing_insufficient_balance":
            raise ValueError(
                f"PyTRIO 登录有效，但工作区余额不足或欠费，未创建训练任务、未更新参数。"
                f"恢复可用额度后重跑同一命令。失败记录：{run / 'result.json'}"
            ) from None
        raise
    except BaseException as error:
        report["status"] = "creation_failed"
        write_json(run / "result.json", report)
        tracker.finish(error)
        raise
    try:
        # 必须使用训练客户端实际提供的 tokenizer，而不是假设本地审计能替代。
        tokenizer = training.get_tokenizer()
        report["chat_template_sha256"] = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        examples, rejected = prepared_rows(rows[:args.max_samples], tokenizer, max_length)
        development, dev_rejected = prepared_rows(dev_rows[:args.max_validation_samples], tokenizer, max_length)
        report.update(rejected=rejected, validation_rejected=dev_rejected,
                      train_rows=len(examples), validation_rows=len(development))
        if not examples or not development or (args.require_complete_epochs and (rejected or dev_rejected)):
            raise ValueError("训练/开发轨迹不完整或超长，请先检查审计结果")
        report["status"] = "running"
        probe = examples[0]
        before = action_nll(training, probe)
        report.update(probe_task_id=probe.task_id, probe_nll_before=before)
        tracker.log({"probe/nll_before": before}, step=0)
        write_json(run / "result.json", report)
        fit(training, examples, development, args, report, run, tracker)
        report["probe_nll_after"] = action_nll(training, probe)
        # 当前4B服务的weights-only恢复返回checkpoint not found；已实测完整状态恢复可用。
        restored = client.create_lora_training_client(
            base_model=args.base_model, rank=args.lora_rank, seed=args.seed,
            train_mlp=True, train_attn=True, train_unembed=False,
        )
        restored.load_state_with_optimizer(report["state_path"]).result()
        report["restore_method"] = "explicit_load_state_with_optimizer"
        report["probe_nll_reloaded"] = action_nll(restored, probe)
        report["probe_nll_changed"] = not math.isclose(before, report["probe_nll_after"], abs_tol=1e-6, rel_tol=1e-6)
        report["reload_matches"] = math.isclose(report["probe_nll_after"], report["probe_nll_reloaded"], abs_tol=1e-3, rel_tol=1e-3)
        report["status"] = "update_and_reload_verified" if report["probe_nll_changed"] and report["reload_matches"] else "needs_inspection"
        tracker.log({"probe/nll_after": report["probe_nll_after"], "probe/nll_reloaded": report["probe_nll_reloaded"],
                     "checks/reload_matches": int(report["reload_matches"]),
                     "checks/full_epochs_completed": int(report["full_epochs_completed"])}, len(report["steps"]))
        write_json(run / "result.json", report)
        if report["status"] == "needs_inspection":
            raise ValueError("更新或重载检查未通过，不能视为完成训练验证")
    except BaseException as error:
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        write_json(run / "result.json", report)
        tracker.finish(error)
        raise
    else:
        tracker.finish()
    print(f"训练执行报告：{run / 'result.json'}")
    print("下一步：用保存的 sampler 权重在开发任务执行购物，训练 NLL 不能代替任务评测。")


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    options, _ = preliminary.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "preflight", "train", "check-logging"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tokenizer", help="仅 audit 使用；可指向已下载的本地 tokenizer")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-validation-samples", type=int, default=16)
    parser.add_argument("--require-complete-epochs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=.03)
    parser.add_argument("--train-price-per-million", type=float, default=4.54)
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline", "disabled"], default="disabled")
    parser.add_argument("--swanlab-project", default="beyond-terminal-reward")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=42)
    if options.config:
        config = json.loads(options.config.read_text())
        unknown = set(config) - {a.dest for a in parser._actions if a.option_strings}
        if unknown:
            parser.error(f"未知配置项：{sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.max_length < 2:
        parser.error("max-length 必须至少为 2")
    return args


def main():
    args = parse_args()
    {"audit": audit, "preflight": preflight, "train": train,
     "check-logging": check_connection}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
