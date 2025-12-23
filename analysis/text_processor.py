import jieba
from snownlp import SnowNLP

def text_segmentation(text):
    """文本分词（中文）"""
    seg_list = jieba.lcut(text)
    return " ".join(seg_list)

def sentiment_analysis(text):
    """情感分析（返回0-1的情感值，越接近1越积极）"""
    s = SnowNLP(text)
    score = round(s.sentiments, 4)
    if score >= 0.7:
        label = "积极"
    elif score <= 0.3:
        label = "消极"
    else:
        label = "中性"
    return {"score": score, "label": label}

def extract_keywords(text, top_k=5):
    """提取关键词（基于词频）"""
    # 分词并过滤停用词（简易版，可扩展停用词表）
    stop_words = {"的", "了", "是", "我", "你", "他", "在", "有", "就", "都"}
    words = [w for w in jieba.lcut(text) if w not in stop_words and len(w) > 1]
    
    # 统计词频
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # 取Top-K
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    keywords = [{"word": w[0], "count": w[1]} for w in sorted_words[:top_k]]
    return keywords
