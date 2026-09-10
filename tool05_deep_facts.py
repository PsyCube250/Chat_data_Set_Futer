"""Tool 05 — Deep fact extraction.

Input: manually attributed external or self-written materials.
Output: structured fact/question pairs with provenance and confidence.

Purification level: 2 / Advanced.
"""

import argparse
import re
import llm_client
import utils

SYSTEM_PROMPT = """你是人物传记素材整理助手，负责从一段文字材料里提取关于"当事人"
（材料中被谈论、被描述、或者材料作者本人如果 source_type 是 self_written 的那个人）
的深层事实性信息——包括重要经历、坚持的价值观、关键人物关系、他人对当事人的印象、
反复出现的主题或隐喻、当事人自我表达中透露出的态度或选择。

用"提问-事实"的对话形式输出每条提取到的信息，模拟"如果有人问当事人一个相关问题，
这段材料能提供的事实依据是什么"。

严格要求：
1. 每条记录必须紧贴材料原文事实，不要脑补、不要过度解读、不要编造材料没有明确提到的内容。
2. 如果材料的表述比较模糊、需要一定推断才能得出，confidence 标为 low 或 medium；
   材料原文清楚写明的，标 high。
3. 一段材料可以提取多条不同角度的事实，不要遗漏重要信息，但也不要为了凑数量把
   同一件事拆成好几条重复的记录。
4. narrator（叙述者）是谁、跟当事人是什么关系，由材料本身提供的元信息决定，
   你不需要也不应该自己判断或改写这两项。
5. 如果材料内容本身是多人对话记录（content 里出现多个不同人名+冒号的说话轮次，
   不只是元信息里标注的那一个 narrator），提取事实时要看清楚这句话到底是谁说的，
   不要把对话里其他人说的话也算成 narrator 说的。如果某条事实明显是材料里
   *另一个人*说的、不是 narrator 本人说的，跳过这条不提取，或在 extracted_fact
   里明确写清楚"（材料中的XX说）"这类归属，不要模糊处理。
"""


def infer_source_type(content: str, speaker: str) -> str:
    """粗略判断材料格式：是长文/单人评论，还是多人对话记录。
    这只是格式分类，不涉及身份归属判断，所以用简单规则做，不需要人工标。
    """
    # 数一下 content 里出现了几个不同的 "名字: " 开头模式（对话记录的典型特征）
    speaker_lines = re.findall(r"^([^\s:：]{2,20})[:：]", content, flags=re.MULTILINE)
    distinct_speakers = set(speaker_lines)
    if len(distinct_speakers) >= 2:
        return "third_party_chat_log"
    return "third_party_essay" if len(content) > 200 else "third_party_comment"


def normalize_materials(raw_data: list) -> list:
    """兼容两种输入格式：
    A) 标准格式：material_id/narrator/narrator_relationship/source_type/raw_text
    B) 你贴的截图那种格式：id/source/speaker/role/target/content
    自动识别并转换成标准格式，B 格式里没有的 source_type 用启发式规则推断
    （只是格式分类，不是身份判断，所以不违反"身份归属必须人工标"的原则）。
    """
    normalized = []
    for item in raw_data:
        if "narrator" in item and "raw_text" in item:
            normalized.append(item)  # 已经是标准格式，原样保留
            continue
        # 格式 B：做字段映射
        content = item.get("content", "")
        speaker = item.get("speaker", "未知")
        material = {
            "material_id": str(item.get("source") or item.get("id") or f"m{len(normalized)}"),
            "narrator": speaker,
            "narrator_relationship": item.get("role", "关系未标注"),
            "source_type": infer_source_type(content, speaker),
            "raw_text": content,
        }
        # target 字段（材料是不是直接关于当事人）作为额外参考信息保留，
        # 不影响提取逻辑，但如果是"间接"提及，值得让人知道这条材料相关性弱一些
        if item.get("target"):
            material["about_target_note"] = item["target"]
        normalized.append(material)
    return normalized

USER_PROMPT_TEMPLATE = """材料元信息：
叙述者: {narrator}
叙述者与当事人的关系: {narrator_relationship}
材料类型: {source_type}

材料原文：
{raw_text}

请提取这段材料里的深层事实信息，按此 JSON 格式返回：
{{"facts": [{{"question": "...", "extracted_fact": "...", "confidence": "high/medium/low"}}]}}
"""


def extract_from_material(material: dict) -> list:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        narrator=material["narrator"],
        narrator_relationship=material["narrator_relationship"],
        source_type=material["source_type"],
        raw_text=material["raw_text"],
    )
    result = llm_client.chat_json(SYSTEM_PROMPT, user_prompt)
    facts = result.get("facts", [])
    for f in facts:
        f["narrator"] = material["narrator"]
        f["narrator_relationship"] = material["narrator_relationship"]
        f["source_type"] = material["source_type"]
        f["material_id"] = material["material_id"]
    return facts


def run(input_path: str, output_path: str):
    materials = utils.load_json(input_path)
    all_facts = []
    for i, material in enumerate(materials, 1):
        print(f"[{i}/{len(materials)}] 处理材料: {material['material_id']} (叙述者: {material['narrator']})")
        try:
            facts = extract_from_material(material)
            all_facts.extend(facts)
            print(f"  提取到 {len(facts)} 条事实")
        except Exception as e:  # noqa: BLE001
            print(f"  [失败，跳过] {e}")

    utils.save_json(all_facts, output_path)
    print(f"\n完成，共提取 {len(all_facts)} 条深层事实，输出到 {output_path}")
    print("提醒：这批数据权重低于 Tool6 的权威访谈结果，整合时如有冲突以 Tool6 为准。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="人工标注好 narrator 等信息的材料文件")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)