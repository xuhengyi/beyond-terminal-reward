"""多轮 SFT 的真实 NLL、数据覆盖、学习率与开发验证。"""

import math
import random
import time

from common import shifted_fields, to_datum, write_json


def weighted_nll(result, examples):
    if len(result.loss_fn_outputs) != len(examples):
        raise ValueError("远程 logprob 数量与输入轨迹数不一致")
    value = 0.0
    for output, example in zip(result.loss_fn_outputs, examples, strict=True):
        weights = shifted_fields(example, len(examples))["weights"]
        logprobs = output["logprobs"].data
        if len(logprobs) != len(weights):
            raise ValueError("logprob 与目标位置没有对齐")
        value -= sum(float(lp) * w for lp, w in zip(logprobs, weights, strict=True) if w > 0)
    if not math.isfinite(value):
        raise ValueError("NLL 非有限值")
    return value


def epoch_batches(examples, batch_size, epochs, seed):
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(examples)
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield epoch, order[start:start + batch_size], start + batch_size >= len(order)


def learning_rate_at(update, total, peak, schedule, warmup_ratio):
    if schedule == "constant":
        return peak
    warmup = math.ceil(total * warmup_ratio)
    # 与 Trainer/LambdaLR 对齐：第一个优化器更新使用 scheduler step=0。
    index = update - 1
    factor = index / max(1, warmup) if index < warmup else (total - index) / max(1, total - warmup)
    # Actor要求学习率严格为正；1e-12近似Trainer首步0，但可通过服务端校验。
    return max(1e-12, peak * max(0.0, factor))


def validation_nll(training, examples, batch_size):
    total = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        result = training.forward([to_datum(x, len(batch)) for x in batch], loss_fn="cross_entropy").result()
        total += weighted_nll(result, batch) * len(batch)
    return total / len(examples)


def fit(training, examples, development, args, report, directory, tracker):
    import pytrio as trio
    total = min(args.steps, args.epochs * math.ceil(len(examples) / args.batch_size))
    report["planned_updates"] = total
    report["epochs"] = []
    seen_tokens = 0
    start_time = time.monotonic()
    for step, (epoch, batch, last) in enumerate(epoch_batches(examples, args.batch_size, args.epochs, args.seed), start=1):
        if step > total:
            break
        lr = learning_rate_at(step, total, args.learning_rate, args.lr_schedule, args.warmup_ratio)
        output = training.forward_backward([to_datum(x, len(batch)) for x in batch], loss_fn="cross_entropy").result()
        loss = weighted_nll(output, batch)
        adam = trio.AdamParams(learning_rate=lr, beta1=.9, beta2=.999, eps=1e-8,
                              weight_decay=0., grad_clip_norm=1.)
        training.optim_step(adam).result()
        inputs = sum(len(x.tokens) - 1 for x in batch)
        seen_tokens += inputs
        entry = {"step": step, "epoch": epoch, "task_ids": [x.task_id for x in batch],
                 "loss": loss, "learning_rate": lr, "input_tokens": inputs,
                 "assistant_tokens": sum(x.action_tokens for x in batch), "metrics": output.metrics}
        report["steps"].append(entry)
        report["training_executed"] = True
        write_json(directory / "result.json", report)
        tracker.log({"train/loss": loss, "train/epoch": epoch, "train/learning_rate": lr,
                     "train/input_tokens_total": seen_tokens,
                     "train/assistant_tokens": entry["assistant_tokens"],
                     "train/elapsed_seconds": time.monotonic() - start_time,
                     "cost/train_estimated_yuan": seen_tokens * args.train_price_per_million / 1e6}, step)
        print(f"epoch={epoch} step={step}/{total} loss={loss:.6f} lr={lr:.3g}", flush=True)
        if last or step == total:
            record = {"epoch": epoch, "step": step, "complete_epoch": last}
            if development:
                record["validation_nll"] = validation_nll(training, development, args.batch_size)
                tracker.log({"validation/nll": record["validation_nll"]}, step)
            prefix = f"shopping-{directory.name}-epoch{epoch}-step{step}"
            state = training.save_state(name=prefix + "-state").result()
            report["state_path"] = state.path
            write_json(directory / "result.json", report)
            sampler = training.save_weights_for_sampler(name=prefix + "-sampler").result()
            report["sampler_path"] = sampler.path
            record.update(state_path=state.path, sampler_path=sampler.path)
            report["epochs"].append(record)
            candidates = [x for x in report["epochs"] if x.get("complete_epoch") and "validation_nll" in x]
            if candidates:
                best = min(candidates, key=lambda x: x["validation_nll"])
                report["best_development_epoch"] = best["epoch"]
                report["best_development_state_path"] = best["state_path"]
                report["best_development_sampler_path"] = best["sampler_path"]
            write_json(directory / "result.json", report)
    report["completed_full_epochs"] = sum(x["complete_epoch"] for x in report["epochs"])
    report["full_epochs_completed"] = report["completed_full_epochs"] == args.epochs
