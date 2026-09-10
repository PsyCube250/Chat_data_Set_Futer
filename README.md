# ChatDataSetFuter

一个用于**聊天数据提纯 → 训练数据构建**的多工具 Agent 工作流。

目标很简单：把一个或多个 JSON 格式的聊天集丢进来，按需要逐级筛选、清洗、标注、审核，最后得到可以继续注册到 JupyterLab / LLaMA-Factory `dataset_info.json` 并直接用于 LoRA SFT 的数据。

## Pipeline

```text
Raw Chat JSON
     │
     ▼
┌──────────── Foundation ────────────┐
│ Tool 01  Preprocess                │
│ Tool 02  Classify                  │
│ Tool 03  Human Calibration        │
└────────────────────────────────────┘
     │
     ▼
┌──────────── Advanced ──────────────┐
│ Tool 04  Integrate / Calibrate     │
│ Tool 05  Deep Facts                │
│ Tool 05.5 Interview Context        │
│ Tool 06  Authoritative Interview  │
└────────────────────────────────────┘
     │
     ▼
┌──────────── Intensive ─────────────┐
│ Tool 07  Style-aware Synthesis    │
│ Tool 08  SFT Review                │
│ Tool 09  RAG Review                │
│ Tool 10  Balanced Final Filter    │
└────────────────────────────────────┘
     │
     ▼
LLaMA-Factory / LoRA-ready dataset
```

### 提纯强度

| 等级 | Tools | 用途 |
|---|---|---|
| **Level 1 — Foundation** | 01–03 | 基础清洗、分类、人工校准 |
| **Level 2 — Advanced** | 04–06 | 分布校准、深层事实、人工确认 |
| **Level 3 — Intensive** | 07–10 | 风格合成、专业审核、RAG 审核、最终配额筛选 |

不是每次都需要跑满 10 个工具。最简单的场景可以只使用 **01 → 02 → 03 → 04**。

## 输入

支持一份或多份聊天 JSON。项目内部统一成：

```json
[
  {
    "conversation_id": "c001",
    "messages": [
      {
        "speaker": "target",
        "text": "你下午排练吗",
        "timestamp": ""
      },
      {
        "speaker": "other",
        "text": "还不知道",
        "timestamp": ""
      }
    ]
  }
]
```

`target` 是需要被学习/克隆的说话人，其他人使用 `other`。

## Quick Start

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your_key"

python main.py preprocess \
  --input data/raw.json \
  --output data/preprocessed.json

python main.py classify \
  --input data/preprocessed.json \
  --output-dir data/

python main.py label \
  --candidates data/candidates.json \
  --output data/labels.json

python main.py integrate \
  --stats data/stats.json \
  --labels data/labels.json \
  --classified data/classified.json \
  --output data/final_dataset.json
```

最终可以用：

```bash
python export_llamafactory.py \
  --input data/final_dataset.json \
  --output data/final_llamafactory.json
```

然后把输出注册到 LLaMA-Factory / JupyterLab 的 `dataset_info.json` 中。`export_llamafactory.py` 是最终导出工具，不属于提纯强度分级。

## Data examples

`data/examples/` 里放的是**完全虚构的示例**，不包含真实聊天内容，用来说明各 Tool 的输出格式。

## Notes

你需要根据自己的数据修改：

- `utils.py` 中的输入格式适配
- `taxonomy.json` 中的标签体系
- `config.py` 中的批量大小和并发数
- 各 Tool 的 prompt / 审核规则

真实数据不要直接提交到 GitHub。
