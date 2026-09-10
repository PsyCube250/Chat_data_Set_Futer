"""Tool 01 — Preprocess / privacy cleanup.

Input: raw or normalized chat JSON.
Output: cleaned conversations with optional context summaries.

Purification level: 1 / Foundation.
"""

import argparse
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import llm_client
import utils

REGEX_PATTERNS = {
    "[手机号]": re.compile(r"1[3-9]\d{9}"),
    "[身份证号]": re.compile(r"\d{17}[\dXx]"),
    "[邮箱]": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
}


def regex_desensitize(text: str) -> str:
    for placeholder, pattern in REGEX_PATTERNS.items():
        text = pattern.sub(placeholder, text)
    return text


SYSTEM_PROMPT = """你是聊天语料预处理助手。给定一批对话（每条消息前标了 [下标][speaker]），
你需要做两件事：

1. 挑出 speaker 为 "other" 的消息里，包含以下情况之一的：
   a) 出现人名/具体地名/单位名等可能识别第三方真实身份的信息 —— 替换成 [人名]/[地名]/[单位]
   b) 转述/引用了别的第三方说的话（比如"我妈说了..."）—— 把被转述的具体内容替换为 [转述内容已省略]，
      只保留 other 自己的态度反应
   只有真的命中以上情况的消息才需要输出，完全没问题的消息不要输出（不需要的比需要的多得多）。
   speaker 为 "target" 的消息一般不用管，除非里面有明显遗漏的强隐私信息（手机号/身份证号等）。

2. 给每通对话生成一句不超过30字的背景摘要。

务必只返回真正需要修改的消息，不要把没改动的消息也列出来，这是为了控制输出长度。
"""

USER_PROMPT_TEMPLATE = """请处理下面这批对话，按此 JSON 格式返回：
{{"结果": [
  {{"conversation_id": "...", "context_summary": "...",
    "edits": {{"下标": "修改后的文本", "下标": "修改后的文本"}}}}
]}}
"edits" 的 key 是消息下标（字符串形式的数字），value 是修改后的完整文本。
没有任何消息需要改的对话，"edits" 给空对象 {{}} 即可，但 "context_summary" 仍要给。

对话数据：
{payload}
"""


def process_batch(batch: list) -> list:
    payload_lines = []
    for conv in batch:
        lines = [f'conversation_id={conv["conversation_id"]}']
        for idx, m in enumerate(conv["messages"]):
            lines.append(f'[{idx}][{m["speaker"]}] {m["text"]}')
        payload_lines.append("\n".join(lines))
    user_prompt = USER_PROMPT_TEMPLATE.format(payload="\n\n".join(payload_lines))
    result = llm_client.chat_json(SYSTEM_PROMPT, user_prompt)
    return result.get("结果", [])


def apply_edits(conv: dict, edit_result: dict) -> dict:
    """把模型返回的局部编辑合并回原始对话，产出 tool2 需要的标准格式。"""
    messages = [dict(m) for m in conv["messages"]]  # 深拷贝一份，避免改到原始数据
    edits = edit_result.get("edits", {})
    for idx_str, new_text in edits.items():
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if 0 <= idx < len(messages):
            messages[idx]["text"] = new_text
    return {
        "conversation_id": conv["conversation_id"],
        "messages": [{"speaker": m["speaker"], "text": m["text"]} for m in messages],
        "context_summary": edit_result.get("context_summary", ""),
    }


def run(input_path: str, output_path: str):
    raw = utils.load_json(input_path)
    conversations = utils.normalize_conversations(raw)
    print(f"[开始] 原始 {len(raw)} 条，标准化后有效对话 {len(conversations)} 通", flush=True)

    for conv in conversations:
        for m in conv["messages"]:
            m["text"] = regex_desensitize(m["text"])

    conv_by_id = {c["conversation_id"]: c for c in conversations}

    processed_by_id = {}
    if os.path.exists(output_path):
        existing = utils.load_json(output_path)
        processed_by_id = {r["conversation_id"]: r for r in existing}
        print(f"[续跑] 检测到已有 {len(processed_by_id)} 条处理结果，将跳过", flush=True)

    todo = [c for c in conversations if c["conversation_id"] not in processed_by_id]
    batches = list(utils.chunk_list(todo, config.PREPROCESS_BATCH_SIZE))
    print(f"[待处理] 剩余 {len(todo)} 通，共 {len(batches)} 批，并发数 {config.PREPROCESS_CONCURRENCY}", flush=True)

    lock = threading.Lock()
    SAVE_EVERY = 10
    t0 = time.time()
    done_count = 0

    def handle_batch(batch):
        try:
            return batch, process_batch(batch), None
        except Exception as e:  # noqa: BLE001
            return batch, None, e

    with ThreadPoolExecutor(max_workers=config.PREPROCESS_CONCURRENCY) as executor:
        futures = [executor.submit(handle_batch, b) for b in batches]
        for i, future in enumerate(as_completed(futures), 1):
            batch, results, err = future.result()
            with lock:
                if err is not None:
                    print(f"  [批次失败，跳过，用正则兜底结果] {err}", flush=True)
                    results = []
                returned_ids = {r["conversation_id"] for r in results}
                for r in results:
                    conv = conv_by_id.get(r["conversation_id"])
                    if conv is not None:
                        processed_by_id[r["conversation_id"]] = apply_edits(conv, r)
                for c in batch:
                    if c["conversation_id"] not in returned_ids:
                        # 模型没返回这条（批次失败/漏了）：用正则兜底结果，摘要留空
                        processed_by_id[c["conversation_id"]] = {
                            "conversation_id": c["conversation_id"],
                            "messages": c["messages"],
                            "context_summary": "",
                        }
                done_count += 1
                elapsed = time.time() - t0
                speed = done_count / elapsed if elapsed > 0 else 0
                eta = (len(batches) - done_count) / speed if speed > 0 else 0
                print(
                    f"[进度] {done_count}/{len(batches)} 批 | 已处理 {len(processed_by_id)} 通 "
                    f"| 耗时 {elapsed:.0f}s | 预计剩余 {eta:.0f}s",
                    flush=True,
                )
                if done_count % SAVE_EVERY == 0:
                    utils.save_json(list(processed_by_id.values()), output_path)
                    print(f"  [已落盘] {output_path}", flush=True)

    utils.save_json(list(processed_by_id.values()), output_path)
    print(f"[完成] 共 {len(processed_by_id)} 通，输出到 {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)