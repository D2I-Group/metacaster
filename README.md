<div align="center">

# (EMNLP2026) MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters

[![Paper](https://img.shields.io/badge/EMNLP-2026-b31b1b.svg)](#citation)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Forecasting](https://img.shields.io/badge/Forecasting-Time_Series-2ea44f.svg)
![Agent](https://img.shields.io/badge/Agent-Meta_Harness-8a2be2.svg)

[English](README.md) | [简体中文](README.zh-CN.md)

🔍 [About](#about) · 🧩 [Framework](#framework) · 🚀 [Quick Start](#quick-start) · 📊 [Results](#results) · 🔗 [Citation](#citation)

</div>

<a id="about"></a>
## 🔍 About

MetaCaster is a multi-agent framework for few-shot time-series forecasting. Instead of using a large language model as the forecaster, MetaCaster uses agents to generate task-adaptive training data and train compact, task-specific Forecasters for downstream deployment.

### 🔑 Key Features

- **Few-shot learning for lightweight Forecasters:** MetaCaster addresses lightweight Forecaster learning under data scarcity, training a task-specific, deployable forecasting model end to end from only a few time-series examples and textual context.
- **Meta-harness-optimized multi-agent framework:** Agents act as intermediary engineers rather than forecasters. HPAgent optimizes MGAgent's Harness against downstream forecasting quality, MGAgent generates task-adaptive training data, and FTAgent trains and selects the best lightweight Forecaster.
- **Unified lightweight forecasting library:** LT-Lib brings together 23 state-of-the-art lightweight Forecasters from 2022–2026 behind unified configuration, training, and evaluation interfaces for model development and selection across architectures.
- **High-quality and efficient forecasting:** Experiments across 18 datasets, 23 lightweight Forecasters, and 14 baselines validate MetaCaster's effectiveness. After deployment, only the selected lightweight Forecaster remains, with no Agent or LLM on the inference path.

<a id="framework"></a>
## 🧩 Framework

<div align="center">

| [<img src="docs/assets/framework.png" width="100%" alt="MetaCaster framework">](docs/assets/framework.png) |
|:--:|
| **Figure 1.** MetaCaster's Harness Optimization framework. |

</div>

MetaCaster provides a unified entry point that runs MGAgent and FTAgent sequentially:

```bash
python -m agents.metacaster
```

It contains three agents:

| Component | Role | Entry point |
|---|---|---|
| **MGAgent** | Generates task-adaptive training windows with the trained Harness | `python -m agents.mgagent.agent` |
| **FTAgent** | Trains LT-Lib Forecasters in parallel and selects Top-1 by validation MSE | `python -m agents.ftagent.agent` |
| **HPAgent** | Optimizes the MGAgent Harness against LT-Lib Forecasters | `python -m agents.hpagent.agent` |

The release already includes the final trained Harness. Run HPAgent only when training a new Harness from scratch.

<a id="quick-start"></a>
## 🚀 Quick Start

Requirements: Python 3.12, [`uv`](https://docs.astral.sh/uv/), an LLM provider key, and a CUDA-capable GPU for Forecaster training. First prepare the environments and data from the release root:

```bash
uv sync
uv sync --project lt_lib
cp .env.example .env  # Fill in the models, API key, and data paths
uv run python scripts/download_gift_eval.py --all
uv run python scripts/prepare_gift_eval.py --all

DATASET="your-dataset-name"
INPUT_DIR="data/GIFT-Eval/test/${DATASET}/agent_view_k30"
TEST_FILE="data/GIFT-Eval/test/${DATASET}/test.npy"
```

### 1. Complete workflow: generate data and train a downstream Forecaster

```bash
uv run python -m agents.metacaster \
  --input-dir "$INPUT_DIR" \
  --test "$TEST_FILE" \
  --gpus 0,1,2,3 \
  --tune \
  --output work_dir/metacaster
```

MGAgent output is saved under `work_dir/metacaster/generated/`. The selected model, configuration, and training results are saved under `work_dir/metacaster/forecaster/`. By default, FTAgent trains the 20-model main pool; pass `--all-models` to train all 23 Forecasters or `--models` to choose specific models.

### 2. Generate data only

```bash
uv run python -m agents.mgagent.agent \
  --input-dir "$INPUT_DIR" \
  --output work_dir/generated \
  "Generate task-adaptive training windows"
```

The generated data is saved to `work_dir/generated/dataset.npy`, with its validation report at `work_dir/generated/validation_report.json`.

### 3. Train a downstream Forecaster only

```bash
uv run --project lt_lib python -m agents.ftagent.agent \
  --synthetic /path/to/dataset.npy \
  --input-dir "$INPUT_DIR" \
  --test "$TEST_FILE" \
  --gpus 0,1,2,3 \
  --main-only \
  --tune \
  --output work_dir/forecaster
```

Omit `--main-only` to train all 23 Forecasters, or use `--models` to select specific models. Use `agents.ftagent.predict` to run the selected model directly; see [`lt_lib/README.md`](lt_lib/README.md).

### 4. Train a New Harness from Scratch

```bash
uv run python -m agents.hpagent.agent \
  --gpus 0,1,2,3
```

HPAgent iteratively optimizes MGAgent's Harness and writes run results under `work_dir/runs/`.

## 📁 Repository Structure

```text
agents/                Unified MetaCaster entry point and individual Agent entry points
generation/            MGAgent runtime
harness/               Final trained router and generation skill
optimizer/             Harness Optimization runtime
lt_lib/    LT-Lib: 23 Forecasters and training runtime
scripts/               GIFT-Eval data preparation
eval/                  Generation-quality evaluation
docs/assets/           Framework and results figures
```

See [`lt_lib/README.md`](lt_lib/README.md) for LT-Lib installation and model validation.

<a id="results"></a>
## 📊 Results

<div align="center">

| [<img src="docs/assets/results.png" width="100%" alt="MetaCaster results">](docs/assets/results.png) |
|:--:|
| **Figure 2.** Forecasting MSE on IND and OOD tasks with $K=10, 30, 50$; MetaCaster achieves the best result in 19 of the 30 dataset–shot settings. |

</div>

<a id="citation"></a>
## 🔗 Citation

```bibtex
@inproceedings{shen2026metacaster,
  title     = {MetaCaster: Meta-Harness-Optimized Agent for End-to-End Few-Shot Learning of Lightweight Time Series Forecasters},
  author    = {Shen, ChengAo and Yu, Wenchao and Wu, Fangyu and Song, Dongjin and Tong, Hanghang and Luo, Dongsheng and Cheng, Wei and Chen, Haifeng and Ni, Jingchao},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## 📧 Contact

If you have any questions or concerns, please contact us: cshen9 [at] uh [dot] edu or submit an issue.
