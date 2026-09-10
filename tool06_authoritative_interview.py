"""Tool 06 — Authoritative interview.

Input: Tool 05.5 interview context.
Output: directly confirmed first-person facts.

Purification level: 2 / Advanced.
"""

import argparse
import datetime
import os

import config
import llm_client
import utils


# ============================================================
# 六个访谈维度
# ============================================================

TOOL6_CATEGORIES = [
    {
        "name": "时间线",
        "hint": "了解重要人生经历、成长阶段、转折、选择、失败、成功以及影响人物变化的关键事件。",
        "source_categories": [
            "identity_background",
            "life_experiences",
        ],
    },
    {
        "name": "爱好偏好",
        "hint": "了解真正重要的兴趣、审美、音乐、艺术、娱乐偏好，以及这些兴趣对本人意味着什么。",
        "source_categories": [
            "interests_preferences",
            "abilities_work",
        ],
    },
    {
        "name": "代表作品",
        "hint": "了解本人真正认为具有代表性的作品、项目、创作、成果，以及为什么这些东西重要。",
        "source_categories": [
            "abilities_work",
            "interests_preferences",
            "life_experiences",
        ],
    },
    {
        "name": "未来打算",
        "hint": "了解未来计划、理想、想实现的事情、职业或人生方向，以及这些选择背后的原因。",
        "source_categories": [
            "values_beliefs",
            "abilities_work",
            "life_experiences",
        ],
    },
    {
        "name": "对待不同朋友的风格与态度",
        "hint": "了解本人如何对待朋友、不同关系中的行为差异、边界、信任、照顾别人以及关系变化。",
        "source_categories": [
            "relationships",
            "personality",
            "emotional_patterns",
            "self_image",
        ],
    },
    {
        "name": "信仰",
        "hint": "了解真正重要的人生观、价值观、原则、相信什么、不相信什么，以及这些观念如何影响实际选择。",
        "source_categories": [
            "values_beliefs",
            "self_image",
        ],
    },
]


# ============================================================
# 提问 Prompt
# ============================================================

QUESTION_SYSTEM_PROMPT = """
你正在帮一个人做一场深度人物访谈。

访谈的目标不是得到漂亮的回答，
而是得到真正能够定义这个人的事实、经历、选择、价值观和行为模式。

当前访谈维度：
{category_name}

这个维度大致想了解：
{category_hint}


============================================================
已有材料
============================================================

下面的信息来自 Tool 05.5。

它们只是辅助材料。

其中：

[本人已确认]
    表示已有材料中存在本人明确表达。

[第三方观察]
    表示其他人对这个人的描述。

[待本人确认]
    表示值得本人亲自确认的内容。

[深挖线索]
    表示可能值得进一步追问的方向。

[需要澄清]
    表示材料中存在需要本人解释的不同描述。


绝对不要把第三方观察直接当成事实。

不要假装本人已经确认了第三方说法。


{known_facts_summary}


============================================================
本轮访谈已经问过
============================================================

{qa_history}


============================================================
提问策略
============================================================

优先级：

1. 如果已有材料中存在高价值的第三方观察，
   优先让本人确认或解释。

2. 如果已经知道某个事实，
   不要简单重新问“是不是”。

3. 优先追问：
   为什么？
   怎么发生的？
   当时为什么这么选择？
   后来有没有改变？
   如果重新来一次会不会不同？
   这件事对你意味着什么？

4. 如果发现本人和第三方对同一事情理解不同，
   优先让本人解释这种差异。

5. 不要为了“深度”强行制造冲突。

6. 不要重复已经问过的角度。

7. 不要问已经明确确认、且没有进一步价值的问题。

8. 问题应该尽量具体，
   让用户可以通过一两句话到几句话回答。

9. 不要问空泛的问题，例如：
   “你怎么看待人生？”
   “你觉得自己是什么样的人？”

   除非已有材料提供了非常具体的切入口。

10. 如果当前维度已经有 3-4 个不同且高价值的角度被充分覆盖，
    可以在问题前加：

    [可结束]

    但仍然需要给出一个问题，让用户自行决定是否继续。


============================================================
输出格式
============================================================

只输出问题本身。

不要解释。

不要编号。

不要 Markdown。

不要说“可以问”。

如果需要结束提示：

[可结束] 你的问题
"""


# ============================================================
# Tool 05.5 → 当前维度上下文
# ============================================================

def build_dimension_context(context: dict, category: dict) -> str:
    """
    从 Tool 05.5 的 deep_facts.json 中，
    提取当前访谈维度相关的信息。

    注意：
        不把整个 context 直接塞给 Tool 06。
    """

    source_categories = set(category.get("source_categories", []))

    sections = []

    # --------------------------------------------------------
    # 1. 本人已确认
    # --------------------------------------------------------

    confirmed = context.get("confirmed_facts", [])

    confirmed_lines = []

    for item in confirmed:
        if item.get("category") in source_categories:
            fact = item.get("fact", "")
            if fact:
                confirmed_lines.append(
                    f"- {fact}"
                )

    if confirmed_lines:
        sections.append(
            "【本人已确认】\n" +
            "\n".join(confirmed_lines[:20])
        )

    # --------------------------------------------------------
    # 2. 第三方观察
    # --------------------------------------------------------

    third_party = context.get("third_party_observations", [])

    third_party_lines = []

    for item in third_party:
        if item.get("category") in source_categories:
            fact = item.get("fact", "")
            source_person = item.get("source_person", "")

            if fact:
                if source_person:
                    third_party_lines.append(
                        f"- {fact}（来源：{source_person}）"
                    )
                else:
                    third_party_lines.append(
                        f"- {fact}"
                    )

    if third_party_lines:
        sections.append(
            "【第三方观察】\n" +
            "\n".join(third_party_lines[:20])
        )

    # --------------------------------------------------------
    # 3. 待本人确认
    # --------------------------------------------------------

    pending = context.get("pending_confirmation", [])

    pending_lines = []

    for item in pending:
        if item.get("category") in source_categories:
            fact = item.get("fact", "")
            suggested_angle = item.get("suggested_angle", "")

            if fact:
                line = f"- {fact}"

                if suggested_angle:
                    line += f"\n  建议确认方向：{suggested_angle}"

                pending_lines.append(line)

    if pending_lines:
        sections.append(
            "【待本人确认】\n" +
            "\n".join(pending_lines[:20])
        )

    # --------------------------------------------------------
    # 4. 高价值深挖
    # --------------------------------------------------------

    deep_dive = context.get("high_value_deep_dive", [])

    deep_dive_lines = []

    for item in deep_dive:
        if item.get("category") in source_categories:
            topic = item.get("topic", "")
            evidence = item.get("evidence", "")
            suggested_angle = item.get("suggested_angle", "")

            if not topic:
                continue

            line = f"- {topic}"

            if evidence:
                line += f"\n  证据：{evidence}"

            if suggested_angle:
                line += f"\n  深挖方向：{suggested_angle}"

            deep_dive_lines.append(line)

    if deep_dive_lines:
        sections.append(
            "【高价值深挖线索】\n" +
            "\n".join(deep_dive_lines[:20])
        )

    # --------------------------------------------------------
    # 5. 冲突 / 需要澄清
    # --------------------------------------------------------

    conflicts = context.get("conflicts", [])

    conflict_lines = []

    for item in conflicts:
        if item.get("category") in source_categories:
            description = item.get("description", "")
            suggested_angle = item.get("suggested_angle", "")

            if description:
                line = f"- {description}"

                if suggested_angle:
                    line += f"\n  澄清方向：{suggested_angle}"

                conflict_lines.append(line)

    if conflict_lines:
        sections.append(
            "【需要澄清】\n" +
            "\n".join(conflict_lines[:10])
        )

    if not sections:
        return "（当前维度暂无相关材料，需要通过访谈主动探索。）"

    return "\n\n".join(sections)


# ============================================================
# 生成问题
# ============================================================

def generate_question(
    category: dict,
    context: dict,
    qa_history: list,
) -> str:

    if qa_history:
        history_text = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in qa_history
        )
    else:
        history_text = "（暂无）"

    known_summary = build_dimension_context(
        context,
        category,
    )

    prompt = QUESTION_SYSTEM_PROMPT.format(
        category_name=category["name"],
        category_hint=category["hint"],
        known_facts_summary=known_summary,
        qa_history=history_text,
    )

    question = llm_client.chat_text(
        prompt,
        "请生成下一个问题。",
    )

    return question.strip()


# ============================================================
# 主流程
# ============================================================

def run(
    context_path: str,
    output_path: str,
):

    # --------------------------------------------------------
    # 读取 Tool 05.5
    # --------------------------------------------------------

    context = {}

    if context_path and os.path.exists(context_path):

        context = utils.load_json(context_path)

        if not isinstance(context, dict):
            raise ValueError(
                "Tool 05.5 输出必须是 JSON object，"
                f"实际类型：{type(context).__name__}"
            )

        print(
            "已加载 Tool 05.5 访谈上下文，"
            "作为提问参考（不会直接采信）\n"
        )

    else:

        print(
            "警告：没有找到 Tool 05.5 上下文，"
            "本次访谈将从零开始。\n"
        )

    all_records = []

    print("=" * 60)

    print(
        "权威访谈开始。每个维度里可以连续回答多个问题；"
    )

    print(
        "输入 next 或直接回车（不打字）跳到下一个维度；"
        "输入 q 提前结束并进入确认环节。"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # 六个维度
    # --------------------------------------------------------

    try:

        for category in TOOL6_CATEGORIES:

            print(
                f"\n--- 维度: {category['name']} ---"
            )

            qa_history = []

            while True:

                try:

                    question = generate_question(
                        category,
                        context,
                        qa_history,
                    )

                except Exception as e:

                    print(
                        f"  [问题生成失败，跳过这一轮] {e}"
                    )

                    break

                print(f"\n问: {question}")

                answer = input(
                    "答（回车/next=下一维度, "
                    "q=结束访谈进入确认）> "
                )

                # ------------------------------------------------
                # 提前结束
                # ------------------------------------------------

                if answer.strip().lower() == "q":

                    raise KeyboardInterrupt

                # ------------------------------------------------
                # 下一维度
                # ------------------------------------------------

                if (
                    answer.strip() == ""
                    or answer.strip().lower() == "next"
                ):

                    break

                # ------------------------------------------------
                # 保存原始回答
                # ------------------------------------------------

                record = {
                    "category": category["name"],
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.datetime.now().isoformat(),
                }

                qa_history.append(record)
                all_records.append(record)

    except KeyboardInterrupt:

        print(
            "\n\n已提前结束访谈，进入确认环节。"
        )

    # --------------------------------------------------------
    # 没有回答
    # --------------------------------------------------------

    if not all_records:

        print(
            "没有记录到任何回答，不生成文件。"
        )

        return

    # ========================================================
    # 最终签署
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "访谈结束，以下是本次记录的全部内容，请核对："
    )

    print("=" * 60)

    for record in all_records:

        print(
            f"\n[{record['category']}]"
        )

        print(
            f"问: {record['question']}"
        )

        print(
            f"答: {record['answer']}"
        )

    print("\n" + "=" * 60)

    print(
        "这些内容一旦确认，将作为整个人格数据集里"
        "权重最高的权威事实，"
    )

    print(
        "后续任何工具（包括 Tool5 / Tool5.5 的素材）"
        "跟这里冲突时都以这里为准。"
    )

    confirm = input(
        "\n确认以上内容真实准确、可以正式落定吗？"
        "输入 确认 来完成签署 > "
    )

    # ========================================================
    # 正式签署
    # ========================================================

    if confirm.strip() == "确认":

        output = {
            "confirmed": True,
            "confirmed_at": datetime.datetime.now().isoformat(),
            "records": all_records,
        }

        utils.save_json(
            output,
            output_path,
        )

        print(
            f"\n已正式签署，输出到 {output_path}"
        )

    # ========================================================
    # 草稿
    # ========================================================

    else:

        draft_path = output_path.replace(
            ".json",
            "_draft.json",
        )

        output = {
            "confirmed": False,
            "records": all_records,
        }

        utils.save_json(
            output,
            draft_path,
        )

        print(
            f"\n未确认，已保存为草稿 {draft_path}"
            "（不会进入最终整合，需要重新运行本工具正式签署）"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context",
        default="data/deep_facts.json",
        help="Tool5.5 产出的 deep_facts.json",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    run(
        args.context,
        args.output,
    )