"""Tool 010 — Balanced final filter.

Input: candidate dataset plus optional human-labeled samples.
Output: final balanced dataset using loss-reduction and fidelity pools.

Purification level: 3 / Intensive.
"""

import argparse
import collections
import concurrent.futures
import math
import random

from utils import load_json, save_json, clean_noise

try:
    from llm_client import chat_json
    import config as _config
except Exception:  # noqa: BLE001
    chat_json = None
    _config = None


DEFAULT_CONFIG = {
    "target_size": None,
    "default_keep_fraction": 0.6,
    "loss_ratio": 0.5,
    "fidelity_ratio": 0.5,
    "loss_pool_max_chars": 25,
    "fidelity_pool_min_chars": 8,
    "fidelity_require_unique": True,
    "random_seed": 42,
    # API打分相关
    "use_api": False,
    "api_batch_size": 20,     # 每次调用塞多少条候选一起打分，对应 config.CHUNK_SIZE 的思路
    "api_concurrency": 6,     # 同时几个请求，对应 config.CLASSIFY_CONCURRENCY
}


API_SCORE_SYSTEM_PROMPT = """你在帮我筛选一批聊天记录里的句子，找出真正体现说话人"个人风格/性格特征"的句子，
用于人格克隆的微调数据集。

判断标准：这句话是否带有可辨认的个人表达习惯、独特用词、情绪色彩、思维方式，而不是任何人都可能说的
通用套话（比如"好""OK""是的"这类要排除在高分之外）。纯链接、纯数字、明显复制粘贴的行程单/表格这类
"信息量大但没有人格信号"的内容也不算高分。

对每条句子打 1-10 分（10=极具个人特色，1=毫无个人特色的套话/无关信息），只输出JSON数组，
每个元素: {"idx": 序号, "score": 分数, "reason": "一句话原因"}，跟输入的idx一一对应，不要遗漏，
不要输出其他任何内容。"""


def build_char_bigram_model(outputs: list) -> tuple:
    unigram = collections.Counter()
    bigram = collections.Counter()
    for text in outputs:
        chars = list(text)
        unigram.update(chars)
        for c1, c2 in zip(chars, chars[1:]):
            bigram[(c1, c2)] += 1
    vocab_size = max(len(unigram), 1)
    total_unigrams = sum(unigram.values())
    return unigram, bigram, vocab_size, total_unigrams


def score_avg_nll(text: str, unigram, bigram, vocab_size, total_unigrams) -> float:
    chars = list(text)
    if not chars:
        return 0.0
    p0 = (unigram.get(chars[0], 0) + 1) / (total_unigrams + vocab_size)
    nll = -math.log(p0)
    for c1, c2 in zip(chars, chars[1:]):
        denom = unigram.get(c1, 0) + vocab_size
        num = bigram.get((c1, c2), 0) + 1
        nll += -math.log(num / denom)
    return nll / len(chars)


def load_labeled_outputs(path: str) -> set:
    """加载 Tool 03 人工标注过的样本文本，统一过 clean_noise() 再存，
    保证跟主数据集里同样清洗过的 output 能对上。
    兼容: ["文本",...] / [{"output":...}] / [{"target_text":..., "human_dims":[...]}]（Tool3实际格式）

    注意：Tool3 里 human_dims 可能是空列表（用户标了"0 都不体现"），这种记录
    不代表这句话有人格信号，不应该被强制塞进还原池——这里明确排除掉
    human_dims 存在但为空的记录，只保留"确实被标注出至少一个性格维度"的样本。
    """
    if not path:
        return set()
    raw = load_json(path)
    labeled = set()
    skipped_empty = 0
    for item in raw:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("target_text") or item.get("output")
            # Tool3格式：human_dims字段存在且为空列表 = 人工判定"不体现任何维度"，跳过
            if "human_dims" in item and not item["human_dims"]:
                skipped_empty += 1
                continue
        else:
            text = None
        if text:
            labeled.add(clean_noise(text))
    if skipped_empty:
        print(f"  labeled文件里有 {skipped_empty} 条 human_dims 为空（标注为'不体现'），已排除，不会强制进还原池")
    return labeled


def api_score_batch(batch: list) -> dict:
    """调用DeepSeek对一批候选打"个性化表达强度"分数，返回 {idx: score}。
    失败（没配key/接口错误/JSON解析失败）时返回空dict，调用方要能接受
    这批全部拿不到分、退回统计代理。
    """
    if chat_json is None:
        return {}
    lines = [f'{{"idx": {i}, "text": {text!r}}}' for i, text in batch]
    user_prompt = "候选句子：\n" + "\n".join(lines)
    try:
        result = chat_json(API_SCORE_SYSTEM_PROMPT, user_prompt)
    except Exception as e:  # noqa: BLE001
        print(f"  [API打分失败，这批 {len(batch)} 条退回统计代理] {e}")
        return {}
    scores = {}
    items = result if isinstance(result, list) else result.get("items", [])
    for entry in items:
        try:
            scores[int(entry["idx"])] = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return scores


def api_score_all(candidates: list, cfg: dict) -> dict:
    """并发批量给候选打分，返回 {idx: score}（idx = candidates里的原始下标位置，
    这里用 s['idx'] 即在原数据里的下标，保证跟后面对齐）。
    """
    batches = []
    cur = []
    for s in candidates:
        cur.append((s["idx"], s["text"]))
        if len(cur) >= cfg["api_batch_size"]:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    all_scores = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["api_concurrency"]) as pool:
        for result in pool.map(api_score_batch, batches):
            all_scores.update(result)
    return all_scores


def build_pools(data: list, cfg: dict, labeled_outputs: set):
    outputs = [d.get("output", "") for d in data]
    unigram, bigram, vocab_size, total_unigrams = build_char_bigram_model(outputs)
    out_counter = collections.Counter(outputs)

    scored = []
    for idx, d in enumerate(data):
        raw_text = d.get("output", "")
        cleaned_text = clean_noise(raw_text)
        nll = score_avg_nll(raw_text, unigram, bigram, vocab_size, total_unigrams)
        scored.append({
            "idx": idx,
            "item": d,
            "text": raw_text,
            "len": len(raw_text),
            "nll": nll,
            "count": out_counter[raw_text],
            "is_labeled": cleaned_text in labeled_outputs,
        })

    # ---- 降loss池：短 + 低困惑度，跟API无关，一直用统计 ----
    loss_candidates = [
        s for s in scored
        if s["len"] > 0 and s["len"] <= cfg["loss_pool_max_chars"]
    ]
    loss_candidates.sort(key=lambda s: s["nll"])

    # ---- 还原池：labeled样本无条件优先，剩余配额用统计或API打分排序 ----
    fidelity_pool_labeled = [s for s in scored if s["is_labeled"]]
    fidelity_pool_stat = [
        s for s in scored
        if not s["is_labeled"]
        and s["len"] >= cfg["fidelity_pool_min_chars"]
        and (s["count"] == 1 if cfg["fidelity_require_unique"] else True)
    ]

    if cfg["use_api"]:
        print(f"  正在用API给 {len(fidelity_pool_stat)} 条候选打个性化表达强度分数...")
        api_scores = api_score_all(fidelity_pool_stat, cfg)
        fallback_count = 0
        for s in fidelity_pool_stat:
            if s["idx"] in api_scores:
                s["fidelity_score"] = api_scores[s["idx"]]
            else:
                s["fidelity_score"] = s["nll"]  # 拿不到API分的退回统计代理
                fallback_count += 1
        if fallback_count:
            print(f"  提醒：{fallback_count} 条没拿到API分数，已退回统计代理打分")
        fidelity_pool_stat.sort(key=lambda s: s["fidelity_score"], reverse=True)
    else:
        fidelity_pool_stat.sort(key=lambda s: s["nll"], reverse=True)

    fidelity_candidates = fidelity_pool_labeled + fidelity_pool_stat

    return loss_candidates, fidelity_candidates


def merge_pools(loss_candidates, fidelity_candidates, cfg: dict, total_data_size: int) -> tuple:
    target_size = cfg["target_size"] or int(total_data_size * cfg["default_keep_fraction"])
    loss_quota = int(target_size * cfg["loss_ratio"])
    fidelity_quota = int(target_size * cfg["fidelity_ratio"])

    used_idx = set()
    final = []

    taken_loss = 0
    for s in loss_candidates:
        if taken_loss >= loss_quota:
            break
        if s["idx"] in used_idx:
            continue
        final.append(s)
        used_idx.add(s["idx"])
        taken_loss += 1

    taken_fidelity = 0
    for s in fidelity_candidates:
        if taken_fidelity >= fidelity_quota:
            break
        if s["idx"] in used_idx:
            continue
        final.append(s)
        used_idx.add(s["idx"])
        taken_fidelity += 1

    report = {
        "target_size": target_size,
        "loss_quota": loss_quota,
        "loss_quota_filled": taken_loss,
        "fidelity_quota": fidelity_quota,
        "fidelity_quota_filled": taken_fidelity,
        "final_count": len(final),
    }
    return final, report


def print_report(report: dict, total_data_size: int) -> None:
    print("=" * 60)
    print("Tool 07 (balanced) 筛选报告")
    print("=" * 60)
    print(f"原始样本数:        {total_data_size}")
    print(f"目标数据集大小:      {report['target_size']}")
    print(f"降loss池配额/实际取到: {report['loss_quota']} / {report['loss_quota_filled']}")
    print(f"还原池配额/实际取到: {report['fidelity_quota']} / {report['fidelity_quota_filled']}")
    print(f"最终样本数:        {report['final_count']}")
    if report["loss_quota_filled"] < report["loss_quota"]:
        print("  注意：降loss池候选不够，配额没填满 —— 可以调高 loss_pool_max_chars")
    if report["fidelity_quota_filled"] < report["fidelity_quota"]:
        print("  注意：还原池候选不够，配额没填满 —— 可以调低 fidelity_pool_min_chars"
              " 或关闭 fidelity_require_unique")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Tool 07 balanced: loss与人格还原的双池配额筛选")
    parser.add_argument("--input", default="persona_dataV4.json")
    parser.add_argument("--output", default="persona_dataV4_balanced.json")
    parser.add_argument("--labeled-file", default=None, help="Tool 03 人工标注候选样本文件（可选）")
    parser.add_argument("--target-size", type=int, default=DEFAULT_CONFIG["target_size"])
    parser.add_argument("--loss-ratio", type=float, default=DEFAULT_CONFIG["loss_ratio"])
    parser.add_argument("--fidelity-ratio", type=float, default=DEFAULT_CONFIG["fidelity_ratio"])
    parser.add_argument("--loss-pool-max-chars", type=int, default=DEFAULT_CONFIG["loss_pool_max_chars"])
    parser.add_argument("--fidelity-pool-min-chars", type=int, default=DEFAULT_CONFIG["fidelity_pool_min_chars"])
    parser.add_argument("--fidelity-allow-duplicates", action="store_true",
                         help="默认还原池只要语料里独一份的句子；加这个开关允许重复句也进还原池")
    parser.add_argument("--use-api", action="store_true", default=DEFAULT_CONFIG["use_api"],
                         help="还原池排序改用DeepSeek打分（需要 DEEPSEEK_API_KEY），失败自动退回统计代理")
    parser.add_argument("--api-batch-size", type=int, default=DEFAULT_CONFIG["api_batch_size"])
    parser.add_argument("--api-concurrency", type=int, default=DEFAULT_CONFIG["api_concurrency"])
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        target_size=args.target_size,
        loss_ratio=args.loss_ratio,
        fidelity_ratio=args.fidelity_ratio,
        loss_pool_max_chars=args.loss_pool_max_chars,
        fidelity_pool_min_chars=args.fidelity_pool_min_chars,
        fidelity_require_unique=not args.fidelity_allow_duplicates,
        use_api=args.use_api,
        api_batch_size=args.api_batch_size,
        api_concurrency=args.api_concurrency,
    )

    if cfg["use_api"] and chat_json is None:
        print("警告：找不到 llm_client/config（没跟config.py/llm_client.py放同目录？），--use-api 会被忽略，退回统计代理")
        cfg["use_api"] = False

    if cfg["loss_ratio"] + cfg["fidelity_ratio"] > 1.0001:
        raise ValueError("loss_ratio + fidelity_ratio 不能超过 1")

    data = load_json(args.input)
    labeled_outputs = load_labeled_outputs(args.labeled_file)

    loss_candidates, fidelity_candidates = build_pools(data, cfg, labeled_outputs)
    final_scored, report = merge_pools(loss_candidates, fidelity_candidates, cfg, len(data))

    random.seed(cfg["random_seed"])
    random.shuffle(final_scored)  # 打乱两池顺序，避免训练时前后半程分布突变
    final_data = [s["item"] for s in final_scored]

    print_report(report, len(data))
    save_json(final_data, args.output)
    print(f"\n已写入: {args.output}")


if __name__ == "__main__":
    main()