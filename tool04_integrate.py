"""Tool 04 — Calibrate and integrate.

Input: Tool 02 classifications/statistics + Tool 03 human labels.
Output: calibrated dataset assembled from the full corpus.

Purification level: 2 / Advanced.
"""

import argparse
from collections import defaultdict

import config
import utils

CONFIDENCE_KEEP_THRESHOLD = 0.5  # 分类置信度低于这个值的标签，最终数据集里不保留


def compute_precision(labels: list) -> dict:
    """precision[dim] = 人工标注里，模型猜这个维度、且人工也认可的比例。"""
    guessed = defaultdict(int)
    confirmed = defaultdict(int)
    for item in labels:
        dim = item["model_dim"]
        guessed[dim] += 1
        if dim in item.get("human_dims", []):
            confirmed[dim] += 1
    precision = {}
    for dim, n in guessed.items():
        precision[dim] = round(confirmed[dim] / n, 3) if n else 1.0
    return precision


def compute_adjusted_proportions(stats: dict, precision: dict, taxonomy: dict) -> dict:
    raw_counts = stats.get("dimension_counts", {})
    adjusted = {}
    for dim in taxonomy:
        raw = raw_counts.get(dim, 0)
        p = precision.get(dim)
        if p is None:
            print(f"  [提示] 维度「{dim}」在人工标注里没有校准数据，沿用模型原始统计，建议 Tool3 再补标几条")
            p = 1.0
        adjusted[dim] = raw * p
    total = sum(adjusted.values())
    proportions = {dim: round(v / total, 4) for dim, v in adjusted.items()} if total else {}
    return adjusted, proportions


def build_final_dataset(classified: list, proportions: dict, combo_proportions: dict, target_size):
    # 只保留置信度达标的标签
    filtered = []
    for rec in classified:
        kept_labels = [l for l in rec["labels"] if l["confidence"] >= CONFIDENCE_KEEP_THRESHOLD]
        if kept_labels:
            filtered.append({**rec, "labels": kept_labels})

    if target_size is None:
        return filtered

    selected = {}

    # 第一步：优先保住高频"组合"样本的配额，避免按单维度独立抽样把
    # 天然绑定在一起的组合特征（比如"调皮+温柔"）拆散成两个独立维度分别抽。
    # 只对本来占比较高（>=1%）的组合给配额，长尾组合不单独保配额，留给
    # 第二步的单维度抽样自然覆盖。
    for combo_key, prop in combo_proportions.items():
        if prop < 0.01:
            continue
        dims = combo_key.split(" + ")
        quota = max(1, round(target_size * prop))
        combo_records = [
            r for r in filtered
            if set(dims).issubset({l["dim"] for l in r["labels"]})
        ]
        combo_records.sort(
            key=lambda r: min(l["confidence"] for l in r["labels"] if l["dim"] in dims),
            reverse=True,
        )
        for r in combo_records[:quota]:
            key = (r["conversation_id"], r["exchange_index"])
            selected[key] = r

    # 第二步：按修正后的单维度比例补齐剩余配额（会自然跳过第一步已选中的记录）
    for dim, prop in proportions.items():
        quota = round(target_size * prop)
        dim_records = [r for r in filtered if any(l["dim"] == dim for l in r["labels"])]
        dim_records.sort(key=lambda r: max(l["confidence"] for l in r["labels"] if l["dim"] == dim), reverse=True)
        already = sum(1 for r in dim_records if (r["conversation_id"], r["exchange_index"]) in selected)
        remaining_quota = max(0, quota - already)
        for r in dim_records:
            key = (r["conversation_id"], r["exchange_index"])
            if key in selected:
                continue
            if remaining_quota <= 0:
                break
            selected[key] = r
            remaining_quota -= 1

    result = list(selected.values())
    if len(result) < target_size:
        # 不够的话从剩余记录里随便补一些，保证数据量
        remaining = [r for r in filtered if (r["conversation_id"], r["exchange_index"]) not in selected]
        result.extend(remaining[: target_size - len(result)])
    return result[:target_size] if len(result) > target_size else result


def run(stats_path: str, labels_path: str, classified_path: str, output_path: str):
    stats = utils.load_json(stats_path)
    labels = utils.load_json(labels_path)
    classified = utils.load_json(classified_path)
    taxonomy = utils.load_taxonomy(config.TAXONOMY_PATH)

    precision = compute_precision(labels)
    adjusted_counts, proportions = compute_adjusted_proportions(stats, precision, taxonomy)
    combo_proportions = stats.get("combination_proportions", {})

    final_records = build_final_dataset(classified, proportions, combo_proportions, config.FINAL_DATASET_SIZE)

    output = {
        "meta": {
            "total_exchanges_in_corpus": stats.get("total_exchanges"),
            "final_dataset_size": len(final_records),
            "model_precision_per_dimension": precision,
            "raw_proportions": stats.get("dimension_proportions"),
            "calibrated_proportions": proportions,
            "combination_proportions": combo_proportions,
            "confidence_keep_threshold": CONFIDENCE_KEEP_THRESHOLD,
        },
        "records": final_records,
    }
    utils.save_json(output, output_path)

    print(f"最终数据集：{len(final_records)} 条记录，输出到 {output_path}")
    print("校准后各维度比例（自然比例，未拉平）：")
    for dim, p in sorted(proportions.items(), key=lambda x: -x[1]):
        print(f"  {dim}: {p:.1%}  (模型精确率 {precision.get(dim, '无校准数据')})")
    if combo_proportions:
        print("高频维度组合（已在抽样时优先保留配额）：")
        for combo, p in sorted(combo_proportions.items(), key=lambda x: -x[1])[:10]:
            print(f"  {combo}: {p:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--classified", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.stats, args.labels, args.classified, args.output)
