"""Tool 08 — SFT quality review.

Input: candidate SFT records.
Output: reviewed records plus auditable deletion reasons and progress.

Purification level: 3 / Intensive.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm_client


# ============================================================
# 你主要修改这里
# ============================================================

SYSTEM_PROMPT = """
审核单条SFT聊天数据，只返回JSON对象。

规则：
1. 任意字段含“[引用]”则删除。
2. 任意字段含“[人名]”则删除。

格式：
{"keep": true, "reason": "简短原因"}
或
{"keep": false, "reason": "简短原因"}

禁止
""".strip()

# ============================================================
# 配置
# ============================================================

INPUT_PATH = "data/persona_dataV3.json"

OUTPUT_PATH = "data/persona_dataV4.json"

DELETED_PATH = "data/deleted.json"

PROGRESS_PATH = "data/tool8_progress.json"

# 并发数
MAX_WORKERS = 32

# 每处理多少条保存一次进度
SAVE_EVERY = 1000


# ============================================================
# JSON 工具
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
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
# 单条审核
# ============================================================

def review_one(index, item):
    """
    审核一条数据。

    返回：
    {
        "index": int,
        "keep": bool,
        "reason": str
    }
    """

    # 尽量完整地把原始训练数据交给审核模型
    user_prompt = json.dumps(
        {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "system": item.get("system", "")
        },
        ensure_ascii=False,
        indent=2
    )

    try:
        result = llm_client.chat_json(
            SYSTEM_PROMPT,
            user_prompt
        )

        keep = result.get("keep")



        reason = str(
            result.get("reason", "")
        ).strip()

        return {
            "index": index,
            "keep": keep,
            "reason": reason
        }

    except Exception as e:
        print(
            f"\n  [LLM失败] index={index}: {e}"
        )

        # API失败绝对不能误删
        return {
            "index": index,
            "keep": None,
            "reason": f"LLM调用失败，未审核：{e}"
        }


# ============================================================
# 进度
# ============================================================

def load_progress():
    if not os.path.exists(PROGRESS_PATH):
        return {}

    try:
        data = load_json(PROGRESS_PATH)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception as e:
        print(f"[警告] 读取进度失败，将重新开始：{e}")
        return {}


def save_progress(results):
    save_json(PROGRESS_PATH, results)


# ============================================================
# 最终生成 V3 + deleted
# ============================================================

def build_outputs(data, results):
    v3 = []
    deleted = []

    for index, item in enumerate(data):

        result = results.get(str(index))

        # 没有审核结果
        if result is None:
            continue

        # API失败 / 未完成
        if result.get("keep") is None:
            continue

        if result["keep"]:
            v3.append(item)

        else:
            deleted_item = dict(item)

            deleted_item["_tool8_index"] = index
            deleted_item["_delete_reason"] = result.get(
                "reason",
                ""
            )

            deleted.append(deleted_item)

    return v3, deleted


# ============================================================
# 主程序
# ============================================================

def run():

    print("=" * 70)
    print("Tool 08：SFT 数据专业审核器")
    print("=" * 70)

    # --------------------------------------------------------
    # 读取 V2
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_PATH}"
        )

    data = load_json(INPUT_PATH)

    if not isinstance(data, list):
        raise ValueError(
            "persona_dataV2.json 必须是 JSON 数组"
        )

    total = len(data)

    print(f"\n输入数据：{INPUT_PATH}")
    print(f"总数据量：{total}")

    print("\n输出：")
    print(f"  保留 → {OUTPUT_PATH}")
    print(f"  删除 → {DELETED_PATH}")
    print(f"  进度 → {PROGRESS_PATH}")

    print(f"\n并发数：{MAX_WORKERS}")

    # --------------------------------------------------------
    # 读取历史进度
    # --------------------------------------------------------

    results = load_progress()

    completed_before = len(results)

    if completed_before:
        print(
            f"\n检测到已有进度："
            f"{completed_before}/{total}"
        )

        print("将跳过已经审核完成的数据。")

    else:
        print("\n没有历史进度，从第 1 条开始。")

    # --------------------------------------------------------
    # 待审核
    # --------------------------------------------------------

    pending = []

    for index, item in enumerate(data):

        if str(index) not in results:
            pending.append(
                (index, item)
            )

    print(
        f"本次需要审核：{len(pending)} 条"
    )

    if not pending:
        print("\n所有数据已经审核完成。")
    else:

        print("\n开始审核……")
        print("按 Ctrl+C 可以随时停止。")
        print("原始 persona_dataV2.json 不会被修改。\n")

        # ----------------------------------------------------
        # 多线程
        # ----------------------------------------------------

        executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        )

        futures = {
            executor.submit(
                review_one,
                index,
                item
            ): index

            for index, item in pending
        }

        processed_this_run = 0

        try:

            for future in as_completed(futures):

                index = futures[future]

                try:
                    result = future.result()

                except Exception as e:

                    result = {
                        "index": index,
                        "keep": None,
                        "reason": f"程序异常，未审核：{e}"
                    }

                results[str(index)] = result

                processed_this_run += 1

                # ------------------------------------------------
                # 统计
                # ------------------------------------------------

                keep_count = sum(
                    1
                    for r in results.values()
                    if r.get("keep") is True
                )

                delete_count = sum(
                    1
                    for r in results.values()
                    if r.get("keep") is False
                )

                failed_count = sum(
                    1
                    for r in results.values()
                    if r.get("keep") is None
                )

                done_count = (
                    keep_count
                    + delete_count
                    + failed_count
                )

                # ------------------------------------------------
                # 终端显示
                # ------------------------------------------------

                status = result.get("keep")

                if status is True:
                    symbol = "保留"
                elif status is False:
                    symbol = "删除"
                else:
                    symbol = "失败"

                print(
                    f"[{done_count}/{total}] "
                    f"index={index} "
                    f"{symbol} "
                    f"| 保留 {keep_count} "
                    f"| 删除 {delete_count} "
                    f"| 失败 {failed_count}"
                )

                reason = result.get("reason", "")

                if reason:
                    print(
                        f"    └─ {reason}"
                    )

                # ------------------------------------------------
                # 定期保存进度
                # ------------------------------------------------

                if (
                    processed_this_run
                    % SAVE_EVERY
                    == 0
                ):
                    save_progress(results)

                    print(
                        f"    [进度已保存："
                        f"{done_count}/{total}]"
                    )

        except KeyboardInterrupt:

            print("\n")
            print("=" * 70)
            print("收到 Ctrl+C，立即保存当前进度。")
            print("=" * 70)

            save_progress(results)

            print(
                f"\n已保存："
                f"{len(results)}/{total}"
            )

            print(
                f"进度文件：{PROGRESS_PATH}"
            )

            print(
                "\n原始 V2 没有被修改。"
            )

            print(
                "下次重新运行 Tool8 会从未完成的位置继续。"
            )

            executor.shutdown(
                wait=False,
                cancel_futures=True
            )

            return

        finally:

            # 正常情况下等待线程结束
            if not pending:
                executor.shutdown(
                    wait=True
                )

    # --------------------------------------------------------
    # 保存最终进度
    # --------------------------------------------------------

    save_progress(results)

    # --------------------------------------------------------
    # 生成 V3 + deleted
    # --------------------------------------------------------

    v3, deleted = build_outputs(
        data,
        results
    )

    save_json(
        OUTPUT_PATH,
        v3
    )

    save_json(
        DELETED_PATH,
        deleted
    )

    # --------------------------------------------------------
    # 最终统计
    # --------------------------------------------------------

    reviewed = len(v3) + len(deleted)

    failed = total - reviewed

    print("\n")
    print("=" * 70)
    print("Tool 08 审核完成")
    print("=" * 70)

    print(f"\n原始 V2：      {total}")
    print(f"审核完成：     {reviewed}")
    print(f"保留 → V3：    {len(v3)}")
    print(f"删除 → deleted：{len(deleted)}")
    print(f"未完成/失败：   {failed}")

    print("\n文件：")
    print(f"  V3：      {OUTPUT_PATH}")
    print(f"  Deleted： {DELETED_PATH}")
    print(f"  Progress：{PROGRESS_PATH}")

    print("\nV2 原文件没有修改。")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    run()