# Beyond Terminal Reward

在 ShopSimulator 中训练长程购物 Agent。基于 Qwen3.5-4B，使用 PyTRIO 运行 LoRA SFT，并将上游 TRACE 逐轮信用分配接入在线 GRPO。训练侧记录动作级优势、参考模型评分和采样用量。评测侧使用环境的 Reward v3 和冻结的 Final-200 Clean 任务集。

## 运行准备

需要 Linux 或 WSL、Git、uv 以及具备对应模型权限的 PyTRIO 账户。从本仓库根目录执行：

~~~bash
uv sync --locked --no-dev
git clone https://github.com/YYHDBL/shopping-grpo-longhorizon.git vendor/shopping-grpo-longhorizon
git -C vendor/shopping-grpo-longhorizon checkout --detach dc6c9af8e48c8e6325101b39d1d20060f69e4221
git -C vendor/shopping-grpo-longhorizon apply ../../patches/shopping-grpo-runtime-manifest.patch
~~~

按上游安装说明准备 ShopSimulator。上游安装脚本同时安装上游自己的训练依赖。若只运行本仓库的 PyTRIO 实验，可用以下命令单独安装购物环境和构建索引：

~~~bash
(
  set -e
  cd vendor/shopping-grpo-longhorizon/environments/ShopSimulator
  uv venv --python 3.10 .venv-shopsim
  uv pip install --python .venv-shopsim/bin/python -r shop_env/requirements.txt
  cd shop_env
  if [ ! -f data/items_eval_train.json ]; then
    gzip -cd data/fine_items_eval_train_all.json.gz > data/items_eval_train.json.preparing
    echo "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f  data/items_eval_train.json.preparing" | sha256sum -c -
    mv data/items_eval_train.json.preparing data/items_eval_train.json
  fi
  echo "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f  data/items_eval_train.json" | sha256sum -c -
  PYTHONPATH=. ../.venv-shopsim/bin/python scripts/build_index.py
)
uv run --no-dev trio login
~~~

## 训练与评测入口

先使用预检确认平台可用，再审计全量 SFT 数据。训练和评测会产生云端费用。示例先运行小规模 TRACE 试训。各命令的路径参数请使用本次运行实际生成的结果文件。

~~~bash
uv run --no-dev python experiments/shopping-sft/02_sft.py preflight
uv run --no-dev python experiments/shopping-sft/02_sft.py audit --config configs/pytrio-sft-full.json
uv run --no-dev python experiments/shopping-sft/02_sft.py train --config configs/pytrio-sft-full.json --swanlab-mode disabled

uv run --no-dev python experiments/shopping-trace/03_train_pilot.py \
  --run-result path/to/sft/result.json --updates 3 --swanlab-mode local

uv run --no-dev python experiments/shopping-sft/04_evaluate_pilot.py \
  --run-result path/to/sft/result.json --model-kind sft --tasks 5
~~~

TRACE 的正式 Final-200 评测入口接受已完成的 100 步运行报告。传入 `--model-kind trace-grpo --split final --allow-blind-final --tasks 200 --budget-yuan 30`，并用 `--run-result` 指向对应的训练报告。环境由各实验入口在本机创建和关闭。SwanLab 同步入口为 `experiments/shopping-sft/05_sync_evaluation.py`。
