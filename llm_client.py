"""DeepSeek API 的轻量封装（OpenAI 兼容接口）。

所有工具都应该通过这里的 chat_json / chat_text 调用模型，
不要在各个 tool 文件里各自 new 一个 client。
"""
import json
import time

from openai import OpenAI

import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "没有找到 DEEPSEEK_API_KEY，请先 export DEEPSEEK_API_KEY=你的key"
            )
        _client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _client


def chat_text(system_prompt: str, user_prompt: str, retries: int = 3, max_tokens: int = 8192) -> str:
    """普通文本调用，返回模型原始回复文本。"""
    client = get_client()
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                # 输出被截断了——不是错误，但调用方需要知道，否则截断的JSON会解析失败
                print(f"  [警告] 输出被截断（触达 max_tokens={max_tokens}），本批次内容可能偏大，考虑调小 batch size")
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  [LLM 调用失败，第 {attempt + 1} 次重试，{wait}s 后重试] {e}")
            time.sleep(wait)
    raise RuntimeError(f"LLM 调用连续失败: {last_err}")


def chat_json(system_prompt: str, user_prompt: str, retries: int = 3) -> dict:
    """要求模型只返回 JSON，自动解析。解析失败会重试。"""
    strict_system = (
        system_prompt
        + "\n\n重要：只输出合法 JSON，不要任何解释、前后缀、markdown 代码块标记。"
    )
    last_err = None
    for attempt in range(retries):
        raw = chat_text(strict_system, user_prompt, retries=1)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  [JSON 解析失败，第 {attempt + 1} 次重试] {e}\n原始返回: {raw[:300]}")
    raise RuntimeError(f"模型没有返回合法 JSON: {last_err}")