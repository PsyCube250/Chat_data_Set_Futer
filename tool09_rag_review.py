"""Tool 09 — RAG quality review.

Input: question/answer or fact records.
Output: reusable RAG records, deletions, and optional splits.

Purification level: 3 / Intensive.
"""

import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm_client


# ============================================================
# 路径配置
# ============================================================

INPUT_FILE = "data/interview_RAG.json"
OUTPUT_FILE = "data/RAG_final.json"
DELETED_FILE = "data/deleted.json"
PROGRESS_FILE = "data/progress.json"


# ============================================================
# 并发配置
# ============================================================

MAX_WORKERS = 8


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = r"""
你是一名专业的 SFT / RAG 数据审核员。

你的任务是审核输入的数据是否适合作为高质量的训练数据或 RAG 知识数据。

你需要重点判断：

1. 数据是否具有明确的信息价值。
2. question 是否清晰、完整、自然。
3. answer 是否真正回答了 question。
4. answer 是否存在明显错误、幻觉、胡编乱造。
5. 数据是否存在严重的上下文缺失。
6. 数据是否只是闲聊、寒暄、无意义内容。
7. 数据是否包含大量重复、低价值或无法复用的信息。
8. 如果一条数据实际上包含多个独立且有价值的问题，应当考虑拆分。
9. 如果数据可以直接保留，则不要无意义地修改。
10. 不要为了“看起来更专业”而过度修改原始内容。

特别注意：

- 不要因为数据比较口语化就直接删除。
- 不要因为 question 比较短就直接删除。
- 不要擅自添加原始数据中不存在的事实。
- 不要凭空补充答案。
- 如果只是表达方式不够完美，但核心信息完整且有价值，可以保留。
- 如果一条数据包含多个可以独立复用的问题，并且拆分后能够形成更好的训练数据，可以进行拆分。
- 拆分后的每一条数据必须能够独立理解。
- 如果无法合理拆分，就不要强行拆分。

输出格式必须是合法 JSON。

你只能输出以下两种格式之一。

============================================================
格式一：直接保留 / 删除
============================================================

{
    "keep": true,
    "reason": "保留原因"
}

或者：

{
    "keep": false,
    "reason": "删除原因"
}

============================================================
格式二：拆分
============================================================

{
    "keep": true,
    "type": "split",
    "reason": "拆分原因",
    "items": [
        {
            "dimension": "维度",
            "topic": "主题",
            "question": "独立的问题",
            "answer": "对应答案"
        },
        {
            "dimension": "维度",
            "topic": "主题",
            "question": "另一个独立的问题",
            "answer": "对应答案"
        }
    ]
}

要求：

- split 时 items 至少有 2 条。
- 每条 item 都必须独立成立。
- 不得凭空增加事实。
- 如果原始数据只有一个完整问题，不要拆分。
- dimension 和 topic 应尽量保留原始数据已有的信息。
- 如果原始数据没有明确 dimension/topic，可以根据内容做最小程度归纳。
- question 和 answer 必须来自原始信息，不得凭空创造。

最终只输出 JSON，不要输出 Markdown，不要输出 ```json，不要输出解释。
"""


# ============================================================
# 全局状态
# ============================================================

stop_event = threading.Event()

lock = threading.Lock()

completed_count = 0
total_count = 0


# ============================================================
# Ctrl+C
# ============================================================

def handle_sigint(signum, frame):
    """
    第一次 Ctrl+C：
    停止继续提交新的任务，并等待当前任务结束。

    第二次 Ctrl+C：
    直接退出。
    """

    if not stop_event.is_set():
        print("\n")
        print("=" * 60)
        print("收到 Ctrl+C，准备安全停止...")
        print("已经完成的结果会保存到 progress.json")
        print("当前正在运行的请求会尽量等待完成。")
        print("如果需要强制退出，请再次按 Ctrl+C。")
        print("=" * 60)
        stop_event.set()

    else:
        print("\n强制退出。")
        raise KeyboardInterrupt


signal.signal(signal.SIGINT, handle_sigint)


# ============================================================
# JSON 读取
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# JSON 保存
# ============================================================

def save_json(path, data):
    """
    原子写入：
    先写 .tmp，再替换正式文件。
    防止程序中断导致 JSON 文件损坏。
    """

    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(temp_path, path)


# ============================================================
# 构造 User Prompt
# ============================================================

def build_user_prompt(item, index):
    """
    给模型的数据。

    尽量把原始字段完整交给模型，但不改变原数据。
    """

    return f"""
请审核下面第 {index} 条数据。

原始数据：

{json.dumps(item, ensure_ascii=False, indent=2)}

请严格按照 System Prompt 中规定的 JSON 格式输出审核结果。
"""


# ============================================================
# 单条审核
# ============================================================

def review_one(index, item):
    """
    单条数据审核。

    注意：
    这里唯一的 LLM 调用就是：

        llm_client.chat_json()

    Tool 08 不直接创建 OpenAI client。
    """

    global completed_count

    user_prompt = build_user_prompt(item, index)

    try:

        # ====================================================
        # 唯一 API 调用入口
        # ====================================================

        result = llm_client.chat_json(
            SYSTEM_PROMPT,
            user_prompt
        )

        # ====================================================
        # 基础结果校验
        # ====================================================

        if not isinstance(result, dict):
            raise ValueError(
                f"模型返回结果不是 JSON object，而是 {type(result).__name__}"
            )

        keep = result.get("keep")

        if not isinstance(keep, bool):
            raise ValueError(
                f"模型返回的 keep 不是 bool：{keep!r}"
            )

        # ====================================================
        # split 结果校验
        # ====================================================

        if result.get("type") == "split":

            items = result.get("items")

            if not isinstance(items, list):
                raise ValueError(
                    "split 结果缺少合法的 items 数组"
                )

            if len(items) < 2:
                raise ValueError(
                    "split 结果的 items 少于 2 条"
                )

            valid_items = []

            for sub_index, sub_item in enumerate(items):

                if not isinstance(sub_item, dict):
                    raise ValueError(
                        f"split.items[{sub_index}] 不是 object"
                    )

                required_fields = [
                    "dimension",
                    "topic",
                    "question",
                    "answer"
                ]

                for field in required_fields:
                    if field not in sub_item:
                        raise ValueError(
                            f"split.items[{sub_index}] 缺少字段：{field}"
                        )

                valid_items.append({
                    "dimension": sub_item["dimension"],
                    "topic": sub_item["topic"],
                    "question": sub_item["question"],
                    "answer": sub_item["answer"]
                })

            result["items"] = valid_items

        # ====================================================
        # 成功
        # ====================================================

        status = "split" if result.get("type") == "split" else (
            "keep" if keep else "delete"
        )

        with lock:
            completed_count += 1
            print(
                f"[{completed_count}/{total_count}] "
                f"#{index} -> {status}"
            )

        return {
            "index": index,
            "result": result,
            "error": None
        }

    except Exception as e:

        # ====================================================
        # API / JSON / 数据校验失败
        #
        # 非常重要：
        # 失败绝对不能默认 keep=True。
        #
        # keep=None 表示：
        # “这条数据没有完成审核”
        # ====================================================

        error_result = {
            "keep": None,
            "reason": f"审核失败，未做保留/删除判断：{e}"
        }

        with lock:
            completed_count += 1
            print(
                f"[{completed_count}/{total_count}] "
                f"#{index} -> ERROR: {e}"
            )

        return {
            "index": index,
            "result": error_result,
            "error": str(e)
        }


# ============================================================
# 生成最终结果
# ============================================================

def build_outputs(data, results):
    """
    根据审核结果生成：

    RAG_final.json
    deleted.json
    """

    final_data = []
    deleted_data = []

    # 按原始 index 排序
    results = sorted(
        results,
        key=lambda x: x["index"]
    )

    for result_entry in results:

        index = result_entry["index"]
        review = result_entry["result"]

        original_item = data[index]

        keep = review.get("keep")

        # ====================================================
        # 审核失败
        # ====================================================

        if keep is None:

            # 未审核成功的数据不删除。
            # 但是也不直接加入最终数据。
            #
            # 这样避免 API 出错时污染 RAG 数据。

            continue

        # ====================================================
        # 删除
        # ====================================================

        if keep is False:

            deleted_data.append({
                "index": index,
                "original": original_item,
                "reason": review.get(
                    "reason",
                    "模型判定为低质量数据"
                )
            })

            continue

        # ====================================================
        # 保留 + 拆分
        # ====================================================

        if review.get("type") == "split":

            split_items = review.get("items", [])

            for split_item in split_items:

                final_data.append({
                    "dimension": split_item["dimension"],
                    "topic": split_item["topic"],
                    "question": split_item["question"],
                    "answer": split_item["answer"]
                })

            continue

        # ====================================================
        # 普通保留
        # ====================================================

        final_data.append(original_item)

    return final_data, deleted_data


# ============================================================
# 主函数
# ============================================================

def main():

    global total_count
    global completed_count

    print("=" * 70)
    print("Tool 08 - SFT / RAG 数据专业审核器")
    print("=" * 70)

    # ========================================================
    # 检查输入文件
    # ========================================================

    if not os.path.exists(INPUT_FILE):
        print(f"\n[错误] 找不到输入文件：{INPUT_FILE}")
        return

    # ========================================================
    # 读取数据
    # ========================================================

    data = load_json(INPUT_FILE)

    if not isinstance(data, list):
        raise ValueError(
            "interview_RAG.json 顶层必须是 list"
        )

    total_count = len(data)

    print(f"\n输入文件：{INPUT_FILE}")
    print(f"数据总量：{total_count}")
    print(f"并发数：{MAX_WORKERS}")

    # ========================================================
    # 读取已有 progress
    # ========================================================

    results = []

    if os.path.exists(PROGRESS_FILE):

        try:

            progress = load_json(PROGRESS_FILE)

            if isinstance(progress, list):

                results = progress

                print(
                    f"发现已有 progress.json，"
                    f"已完成：{len(results)}"
                )

        except Exception as e:

            print(
                f"[警告] progress.json 读取失败：{e}"
            )
            print("将从头开始。")

    # ========================================================
    # 已完成 index
    # ========================================================

    completed_indexes = {
        item["index"]
        for item in results
        if isinstance(item, dict)
        and "index" in item
    }

    pending = [
        (index, item)
        for index, item in enumerate(data)
        if index not in completed_indexes
    ]

    print(f"待审核：{len(pending)}")

    if not pending:

        print("\n所有数据已经审核完成。")

    else:

        # ====================================================
        # 多线程
        # ====================================================

        executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        )

        futures = {}

        try:

            for index, item in pending:

                if stop_event.is_set():
                    break

                future = executor.submit(
                    review_one,
                    index,
                    item
                )

                futures[future] = index

            # =================================================
            # 收集结果
            # =================================================

            for future in as_completed(futures):

                if stop_event.is_set():

                    # 已提交的任务仍然收集。
                    # 不再提交新任务。
                    pass

                try:

                    result = future.result()

                    results.append(result)

                    # -----------------------------------------
                    # 每完成一条就保存 progress
                    # -----------------------------------------

                    save_json(
                        PROGRESS_FILE,
                        results
                    )

                except Exception as e:

                    index = futures[future]

                    print(
                        f"[错误] #{index} future 执行失败：{e}"
                    )

                # ------------------------------------------------
                # 如果用户 Ctrl+C，继续收集已经提交的任务，
                # 然后退出。
                # ------------------------------------------------

        except KeyboardInterrupt:

            print("\n正在停止...")

        finally:

            executor.shutdown(
                wait=True
            )

            # 最后再保存一次
            save_json(
                PROGRESS_FILE,
                results
            )

    # ========================================================
    # 如果没有完成全部数据
    # ========================================================

    processed_indexes = {
        item["index"]
        for item in results
        if isinstance(item, dict)
        and "index" in item
    }

    unfinished = [
        index
        for index in range(total_count)
        if index not in processed_indexes
    ]

    if unfinished:

        print("\n" + "=" * 70)
        print("本次运行未完成全部数据。")
        print(f"已完成：{len(processed_indexes)} / {total_count}")
        print(f"剩余：{len(unfinished)}")
        print(f"进度文件：{PROGRESS_FILE}")
        print("=" * 70)

        print(
            "\n不会生成新的最终结果，"
            "下次运行会自动继续未完成的数据。"
        )

        return

    # ========================================================
    # 所有数据完成
    # ========================================================

    final_data, deleted_data = build_outputs(
        data,
        results
    )

    # ========================================================
    # 保存最终结果
    # ========================================================

    save_json(
        OUTPUT_FILE,
        final_data
    )

    save_json(
        DELETED_FILE,
        deleted_data
    )

    # ========================================================
    # 统计
    # ========================================================

    split_count = 0
    keep_count = 0
    delete_count = 0
    error_count = 0

    for result_entry in results:

        review = result_entry["result"]
        keep = review.get("keep")

        if keep is None:
            error_count += 1

        elif keep is False:
            delete_count += 1

        elif review.get("type") == "split":
            split_count += 1

        else:
            keep_count += 1

    print("\n")
    print("=" * 70)
    print("Tool 09 审核完成")
    print("=" * 70)

    print(f"原始数据：      {len(data)}")
    print(f"直接保留：      {keep_count}")
    print(f"拆分数据：      {split_count}")
    print(f"删除：          {delete_count}")
    print(f"审核失败：      {error_count}")
    print(f"最终 RAG 数据： {len(final_data)}")

    print("\n输出文件：")
    print(f"  {OUTPUT_FILE}")
    print(f"  {DELETED_FILE}")
    print(f"  {PROGRESS_FILE}")

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()