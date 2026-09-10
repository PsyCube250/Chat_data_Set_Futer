"""公共函数：JSON 读写 + 输入格式适配。"""
import json
import os
import re


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 连续重复的表情/符号（比如刷屏的[流泪][流泪][流泪]...）压缩成最多3个，
# 减少无意义token，不然一条消息可能几百个重复表情把prompt撑爆
_REPEAT_PATTERN = re.compile(r"(\[[^\[\]]{1,6}\])\1{3,}")

MAX_MESSAGE_LEN = 500  # 单条消息超过这个长度就截断（代码/长链接没必要全部喂给LLM）


def clean_noise(text: str) -> str:
    if not text:
        return text
    text = _REPEAT_PATTERN.sub(lambda m: m.group(1) * 3, text)
    if len(text) > MAX_MESSAGE_LEN:
        text = text[:MAX_MESSAGE_LEN] + "...[截断]"
    return text


def normalize_conversations(raw_data) -> list:
    """适配你的实际格式：
    [{"conversations": [{"from": "human"/"gpt", "value": "...", "speaker": "...", "speaker_role": "..."}], "participants": {...}}, ...]
    from=gpt  -> 你自己，标记为 target
    from=human -> 对方，标记为 other
    """
    normalized = []
    for idx, item in enumerate(raw_data):
        msgs = item.get("conversations", [])
        if not msgs:
            continue  # 没有消息内容的空壳直接跳过，不占后续处理名额
        conv_id = f"c{idx:05d}"
        norm_messages = []
        for m in msgs:
            from_field = m.get("from", "")
            speaker = "target" if from_field == "gpt" else "other"
            text = clean_noise(m.get("value", ""))
            if not text.strip():
                continue
            norm_messages.append({"speaker": speaker, "text": text, "timestamp": ""})
        if norm_messages:
            normalized.append({"conversation_id": conv_id, "messages": norm_messages})
    return normalized


def load_taxonomy(path: str) -> dict:
    return load_json(path)


def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
        

