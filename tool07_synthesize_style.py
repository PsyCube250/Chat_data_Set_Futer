"""Tool 07 — Style-aware fact synthesis.

Input: high-confidence third-party observations.
Output: short synthetic dialogue samples grounded in known facts.

Purification level: 3 / Intensive.
"""

import argparse

import config
import llm_client
import utils

# 只采用置信度和相关性都足够高的观察，避免把 pending_confirmation 里没确认的
# 猜测也合成进训练集——那些应该等 Tool6 权威访谈确认或者反驳之后再考虑
MIN_CONFIDENCE = {"high", "medium"}
MIN_RELEVANCE = {"very_high", "high"}

SYSTEM_PROMPT = """你要把一条关于"当事人"的第三方事实观察，改写成一段当事人
自己可能会说的**极短口语化回答**，模拟"如果有人恰好聊到相关话题，当事人会
怎么随口接一句"。

严格要求：
1. 极短，1-2句话，口语、随意，像真实聊天，不要书面语，不要长篇大论。
2. 不能是对这条事实的正式陈述或者自我介绍式的表达，要像日常聊天里
   自然带出来的一句话。
3. 可以体现出一定的自我觉察或者态度（比如认同/不完全认同/轻描淡写这个
   第三方观察），不要机械复述"我是一个XX的人"这种生硬表达。
4. 不要编造事实之外的细节，只基于给定的这条事实来生成。
"""

USER_PROMPT_TEMPLATE = """这是一条关于当事人的第三方观察：
{fact}

请生成：
1. 一个"other"可能会问、或者聊天中恰好会聊到的简短开场（模拟真实聊天场景）
2. 当事人对此的极短口语化回应

按此 JSON 格式返回：
{{"prior_other_text": "...", "target_text": "...", "context_summary": "一句话场景概括"}}
"""


def filter_observations(summary_data: dict) -> list:
    candidates = []
    for obs in summary_data.get("third_party_observations", []):
        if obs.get("confidence") in MIN_CONFIDENCE and obs.get("persona_relevance") in MIN_RELEVANCE:
            candidates.append(obs["fact"])
    return candidates


def synthesize_one(fact: str) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(fact=fact)
    result = llm_client.chat_json(SYSTEM_PROMPT, user_prompt)
    return result


def run(summary_path: str, output_path: str, max_samples: int = 60):
    summary_data = utils.load_json(summary_path)
    facts = filter_observations(summary_data)
    print(f"共 {len(facts)} 条高置信度+高相关性的观察，取前 {max_samples} 条合成")
    facts = facts[:max_samples]

    synthesized = []
    for i, fact in enumerate(facts, 1):
        try:
            result = synthesize_one(fact)
            synthesized.append(
                {
                    "conversation_id": f"synthetic_{i:04d}",
                    "exchange_index": 0,
                    "target_text": result["target_text"],
                    "prior_other_text": result.get("prior_other_text"),
                    "context_summary": result.get("context_summary", ""),
                    "labels": [],  # 合成样本先不参与性格维度统计，避免扭曲 Tool2 的自然分布
                    "is_synthetic": True,
                    "source_fact": fact,
                }
            )
            print(f"  [{i}/{len(facts)}] {result['target_text'][:30]}...")
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(facts)}] 合成失败，跳过: {e}")

    utils.save_json(synthesized, output_path)
    print(f"\n完成，共合成 {len(synthesized)} 条，输出到 {output_path}")
    print("提醒：合并进最终数据集时，这批 is_synthetic=true 的样本占比建议控制在总量的5%以内。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Tool5.5 产出的总结JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=60)
    args = parser.parse_args()
    run(args.summary, args.output, args.max_samples)