"""全局配置。"""
import os

# ---- DeepSeek API ----
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # 需要更强推理可以换 "deepseek-reasoner"

# ---- 路径 ----
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "taxonomy.json")

# ---- Tool 1: 脱敏/摘要 ----
# 每次调用 LLM 打包处理的对话数（避免一条条调用太慢，也避免一次塞太多）
PREPROCESS_BATCH_SIZE = 12
PREPROCESS_CONCURRENCY = 6   # 新增：同时并发几个请求，太大容易触发API限流，先从6试
# ---- Tool 2: 分类统计 ----
CLASSIFY_CONCURRENCY = 6   # 并发请求数，可根据 API 限流调整
# ---- Tool 2: 分类统计 ----
# 每个 chunk 喂给模型的对话数量，语料量大时调小一些防止超上下文
CHUNK_SIZE = 20
# 每个性格维度挑给人工标注的候选样本数（典型 + 边界各占一半左右）
CANDIDATES_PER_DIM = 15

# ---- Tool 3: 人工标注 ----
# 标注目标总条数区间，达到上限后 CLI 会提示可以停了（但不会强制退出）
LABEL_TARGET_MIN = 100
LABEL_TARGET_MAX = 200

# ---- Tool 4: 最终数据集 ----
# 最终数据集希望包含的总样本数上限（None 表示不限，按修正后比例抽满全量语料）
FINAL_DATASET_SIZE = None
# ---- Tool 6: 权威访谈 ----
# 必须覆盖的六个维度，每个维度至少问一个问题，回答直接记录原话，不加工不推测
TOOL6_CATEGORIES = [
    {"name": "时间线", "hint": "重要的人生节点、转折点，发生在什么时候"},
    {"name": "爱好偏好", "hint": "真正喜欢的东西，不是社交场合说说而已的那种"},
    {"name": "代表作品", "hint": "最能代表自己、倾注了心血或灵魂的产出——一首歌/一段文字/一件事"},
    {"name": "未来打算", "hint": "接下来真正想做的事，不是场面话"},
    {"name": "对待不同朋友的风格与态度", "hint": "对不同类型的人/关系，表现出的差异"},
    {"name": "信仰", "hint": "真正相信、坚持的东西，价值观层面的"},
]