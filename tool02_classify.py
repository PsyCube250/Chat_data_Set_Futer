"""Tool 02 — Personality/behavior classification.

Input: Tool 01 output.
Output: per-target labels, distribution statistics, and human-review candidates.

Purification level: 1 / Foundation.
"""

import argparse
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import llm_client
import utils

SYSTEM_PROMPT_TEMPLATE = """你是性格分析助手。下面是一份性格维度定义表：

{taxonomy}

给定一批对话，每条对话包含背景摘要和消息列表。请针对每一条 speaker 为 "target" 的
发言，结合它前面紧邻的 other 发言（如果有）和背景摘要，判断这条 target 发言体现了
定义表里的哪些维度。可以命中 0 个、1 个或多个维度；都不命中就返回空列表。

每个命中的维度给一个 0~1 的置信度（越接近 1 表示体现得越明显/典型，越接近 0.5
表示比较模糊/边界情况）。
"""

USER_PROMPT_TEMPLATE = """请分析下面这批对话，按此 JSON 格式返回：

{{"结果": [
  {{"conversation_id": "...", "exchange_index": 0, "labels": [{{"dim": "调皮", "confidence": 0.85}}]}}
]}}

exchange_index 是这条 target 消息在它所属对话 messages 列表里的下标（从0开始）。

对话数据：

{payload}
"""


def build_exchanges(conversations: list) -> list:
    """从对话里抽出所有 target 发言作为分类单位，附带上下文。"""
    exchanges = []
    for conv in conversations:
        prior_other = None
        for idx, m in enumerate(conv["messages"]):
            if m["speaker"] == "target":
                exchanges.append(
                    {
                        "conversation_id": conv["conversation_id"],
                        "exchange_index": idx,
                        "target_text": m["text"],
                        "prior_other_text": prior_other,
                        "context_summary": conv.get("context_summary", ""),
                    }
                )
            else:
                prior_other = m["text"]
    return exchanges


def format_taxonomy(taxonomy: dict) -> str:
    return "\n".join(f"- {dim}：{desc}" for dim, desc in taxonomy.items())


def classify_conversations_batch(batch: list, taxonomy_str: str) -> list:
    """对一批对话请求 LLM 分类，返回结果列表。"""
    payload_lines = []
    for conv in batch:
        lines = [f'conversation_id={conv["conversation_id"]}', f'背景摘要: {conv.get("context_summary", "")}']
        for idx, m in enumerate(conv["messages"]):
            lines.append(f'[{idx}][{m["speaker"]}] {m["text"]}')
        payload_lines.append("\n".join(lines))

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(taxonomy=taxonomy_str)
    user_prompt = USER_PROMPT_TEMPLATE.format(payload="\n\n".join(payload_lines))
    result = llm_client.chat_json(system_prompt, user_prompt)
    return result.get("结果", [])


def run(input_path: str, output_dir: str):
    conversations = utils.load_json(input_path)
    taxonomy = utils.load_taxonomy(config.TAXONOMY_PATH)
    taxonomy_str = format_taxonomy(taxonomy)

    # exchange 查找表
    exchange_lookup = {}
    for ex in build_exchanges(conversations):
        exchange_lookup[(ex["conversation_id"], ex["exchange_index"])] = ex

    classified = []          # 完整逐条分类结果
    stats = defaultdict(int) # 维度计数
    combo_stats = defaultdict(int) # 维度组合计数
    total_exchanges = len(exchange_lookup)

    COMBO_CONFIDENCE_THRESHOLD = 0.5

    # 分批
    batches = list(utils.chunk_list(conversations, config.CHUNK_SIZE))
    # 并发数：优先用 config.CLASSIFY_CONCURRENCY，若没有则复用 PREPROCESS_CONCURRENCY
    concurrency = getattr(config, "CLASSIFY_CONCURRENCY", config.PREPROCESS_CONCURRENCY)
    print(f"共 {len(batches)} 个批次，并发数 {concurrency}")

    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def process_batch(batch):
        """供线程池调用的包装函数。"""
        try:
            results = classify_conversations_batch(batch, taxonomy_str)
            return batch, results, None
        except Exception as e:
            return batch, None, e

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]

        for future in as_completed(futures):
            batch, results, err = future.result()
            with lock:
                if err is not None:
                    print(f"  [批次失败，跳过该批] {err}")
                    # 该 batch 的对话后续统一补空标签
                else:
                    # 处理正常返回的结果
                    for r in results:
                        key = (r["conversation_id"], r["exchange_index"])
                        ex = exchange_lookup.get(key)
                        if ex is None:
                            continue
                        record = {**ex, "labels": r.get("labels", [])}
                        classified.append(record)

                        # 更新统计
                        for lbl in record["labels"]:
                            stats[lbl["dim"]] += 1

                        confident_dims = sorted(
                            lbl["dim"] for lbl in record["labels"]
                            if lbl["confidence"] >= COMBO_CONFIDENCE_THRESHOLD
                        )
                        if len(confident_dims) >= 2:
                            combo_stats[tuple(confident_dims)] += 1

                completed += 1
                elapsed = time.time() - t0
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (len(batches) - completed) / speed if speed > 0 else 0
                print(f"[进度] {completed}/{len(batches)} 批 | 耗时 {elapsed:.0f}s | 预计剩余 {eta:.0f}s", flush=True)

    # 所有批次完成后，检查是否有 exchange 未被分类（批次失败或模型漏掉）
    classified_keys = {(c["conversation_id"], c["exchange_index"]) for c in classified}
    for key, ex in exchange_lookup.items():
        if key not in classified_keys:
            classified.append({**ex, "labels": []})

    # 统计各维度占比 + 维度组合
    stats_output = {
        "total_exchanges": total_exchanges,
        "dimension_counts": dict(stats),
        "dimension_proportions": {
            dim: round(cnt / total_exchanges, 4) for dim, cnt in stats.items()
        } if total_exchanges else {},
        "combination_counts": {" + ".join(k): v for k, v in combo_stats.items()},
        "combination_proportions": {
            " + ".join(k): round(v / total_exchanges, 4) for k, v in combo_stats.items()
        } if total_exchanges else {},
    }

    # 挑选给人工标注的候选：每个维度里，按置信度排序，
    # 一半取最典型（高置信度）的，一半取边界模糊（置信度接近0.5~0.65）的
    candidates = []
    seen_exchange_keys = set()
    n = config.CANDIDATES_PER_DIM

    for dim in taxonomy:
        hits = [
            (rec, lbl["confidence"])
            for rec in classified
            for lbl in rec["labels"]
            if lbl["dim"] == dim
        ]
        hits.sort(key=lambda x: x[1], reverse=True)
        typical = hits[: n // 2]
        borderline_pool = [h for h in hits if 0.45 <= h[1] <= 0.7]
        borderline = borderline_pool[: n - len(typical)]

        for rec, conf in typical + borderline:
            key = (rec["conversation_id"], rec["exchange_index"])
            if key in seen_exchange_keys:
                continue
            seen_exchange_keys.add(key)
            candidates.append(
                {
                    "conversation_id": rec["conversation_id"],
                    "exchange_index": rec["exchange_index"],
                    "target_text": rec["target_text"],
                    "prior_other_text": rec["prior_other_text"],
                    "context_summary": rec["context_summary"],
                    "model_dim": dim,
                    "model_confidence": conf,
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    utils.save_json(classified, os.path.join(output_dir, "classified.json"))
    utils.save_json(stats_output, os.path.join(output_dir, "stats.json"))
    utils.save_json(candidates, os.path.join(output_dir, "candidates.json"))

    print(f"完成：共 {total_exchanges} 条 target 发言，分类产出 {len(classified)} 条记录")
    print(f"挑出 {len(candidates)} 条候选样本给 Tool3 人工标注")
    print(f"维度分布: {stats_output['dimension_proportions']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.input, args.output_dir)