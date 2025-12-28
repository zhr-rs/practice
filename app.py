import jieba
from collections import Counter
import re
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar, Line, Pie
import os
import requests
from bs4 import BeautifulSoup
import glob
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_echarts import st_pyecharts

# ====================== 全局配置 ======================
# 配置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 全局User-Agent（避免重复定义）
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 加载外部停用词表（替代硬编码）
def load_stopwords():
    """加载停用词表"""
    stopwords = set()
    try:
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        st.warning("⚠️ 未找到stopwords.txt，使用默认精简停用词表")
        stopwords = {"的", "了", "是", "这", "那", "在", "和", "就", "都", "也", "还"}
    return stopwords

STOP_WORDS = load_stopwords()

# ====================== 网页爬取函数 ======================
def get_webpage_content(url):
    """爬取指定URL的网页正文（适配fulong_news_content容器）"""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding  # 自动识别编码
        response.raise_for_status()  # 抛出HTTP错误

        soup = BeautifulSoup(response.text, "html.parser")
        # 移除无关标签
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            tag.decompose()

        # 提取核心正文（可根据目标网站修改class名）
        content_div = soup.find("div", class_="fulong_news_content")
        if content_div:
            final_content = content_div.get_text(strip=True, separator="\n")
            # 过滤空行
            final_content = "\n".join([line for line in final_content.split("\n") if line.strip()])
            return final_content
        else:
            # 调试：返回网页前10个div的类名，方便适配其他网站
            all_div_classes = [div.get("class") for div in soup.find_all("div", class_=True)[:10]]
            return f"❌ 未找到'fulong_news_content'容器！\n网页前10个div类名：{all_div_classes}"

    except requests.exceptions.RequestException as e:
        return f"❌ 爬取失败：{str(e)}"
    except Exception as e:
        return f"❌ 解析失败：{str(e)}"

# ====================== 文件保存函数 ======================
def save_content_to_file(content, file_name):
    """保存爬取内容到本地文件"""
    # 获取当前脚本所在目录（避免路径问题）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        st.success(f"✅ {file_name} 保存成功！路径：{file_path}")
    except Exception as e:
        st.error(f"❌ {file_name} 保存失败：{str(e)}")

# ====================== 词频统计+过滤函数 ======================
def get_single_file_word_freq(file_name, min_freq):
    """
    读取单个新闻文件，分词+过滤停用词/低频词，返回前20词频
    :param file_name: 单个新闻文件路径
    :param min_freq: 最小词频阈值
    :return: 排序后的前20词频字典
    """
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            total_text = f.read()
    except Exception as e:
        st.warning(f"⚠️ 读取{file_name}失败：{str(e)}")
        return {}
    
    if not total_text:
        return {}
    
    # 文本清洗：只保留中文
    clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", total_text)
    # 分词 + 过滤停用词/单字
    words = [w for w in jieba.lcut(clean_text) if w not in STOP_WORDS and len(w) > 1]
    # 统计词频 + 过滤低频词
    word_count = Counter(words)
    filtered_words = {word: freq for word, freq in word_count.items() if freq >= min_freq}
    # 取前20并按词频降序排序
    top20_words = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20])
    return top20_words

def get_merged_file_word_freq(file_list, min_freq):
    """
    （保留原有功能）合并所有新闻文件，分词+过滤停用词/低频词，返回前20词频
    :param file_list: 新闻文件列表
    :param min_freq: 最小词频阈值
    :return: 排序后的前20词频字典
    """
    total_text = ""
    # 合并所有新闻文件内容
    for file_name in file_list:
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                total_text += f.read() + "\n"
        except Exception as e:
            st.warning(f"⚠️ 读取{file_name}失败：{str(e)}")
            continue
    
    if not total_text:
        return {}
    
    # 文本清洗：只保留中文
    clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", total_text)
    # 分词 + 过滤停用词/单字
    words = [w for w in jieba.lcut(clean_text) if w not in STOP_WORDS and len(w) > 1]
    # 统计词频 + 过滤低频词
    word_count = Counter(words)
    filtered_words = {word: freq for word, freq in word_count.items() if freq >= min_freq}
    # 取前20并按词频降序排序
    top20_words = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20])
    return top20_words

# ====================== 多图表渲染函数 ======================
def render_chart(chart_type, top20_words, title_suffix=""):
    """
    根据选择的图表类型渲染可视化图形
    :param chart_type: 图表类型
    :param top20_words: 词频字典
    :param title_suffix: 标题后缀（区分不同链接）
    """
    words = list(top20_words.keys())
    freqs = list(top20_words.values())
    
    if not words:
        st.warning(f"⚠️ 过滤后无有效词汇！{title_suffix} 请降低'最小词频'阈值")
        return
    
    # 1. 柱状图（词频前20）
    if chart_type == "柱状图（词频前20）":
        bar = (
            Bar(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", freqs, itemstyle_opts=opts.ItemStyleOpts(color="#1890ff"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"新闻文本词频前20 - 柱状图 {title_suffix}", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                yaxis_opts=opts.AxisOpts(name="出现次数"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow")
            )
        )
        st_pyecharts(bar)
    
    # 2. 横向柱状图（词频前20）
    elif chart_type == "横向柱状图（词频前20）":
        bar = (
            Bar(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", freqs, itemstyle_opts=opts.ItemStyleOpts(color="#52c41a"))
            .reversal_axis()  # 反转轴，转为横向
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"新闻文本词频前20 - 横向柱状图 {title_suffix}", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
                yaxis_opts=opts.AxisOpts(name="词汇"),
                xaxis_opts=opts.AxisOpts(name="出现次数"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow")
            )
        )
        st_pyecharts(bar)
    
    # 3. 折线图（词频趋势）
    elif chart_type == "折线图（词频趋势）":
        line = (
            Line(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", freqs, is_smooth=True, itemstyle_opts=opts.ItemStyleOpts(color="#f5222d"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"新闻文本词频前20 - 折线图 {title_suffix}", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
                yaxis_opts=opts.AxisOpts(name="出现次数"),
                tooltip_opts=opts.TooltipOpts(trigger="axis")
            )
        )
        st_pyecharts(line)
    
    # 4. 饼图（词频占比，取前10避免重叠）
    elif chart_type == "饼图（词频占比）":
        pie = (
            Pie(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add(
                series_name="词频占比",
                data_pair=list(zip(words[:10], freqs[:10])),
                radius=["30%", "75%"],
                center=["50%", "50%"]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"新闻文本词频前10 - 饼图 {title_suffix}", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
                legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%")
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(formatter="{b}: {c}次 ({d}%)")
            )
        )
        st_pyecharts(pie)
    
    # 5. 面积图（词频累积，matplotlib）
    elif chart_type == "面积图（词频累积）":
        fig, ax = plt.subplots(figsize=(12, 6))
        # 绘制面积图
        ax.fill_between(words, freqs, color="#fa8c16", alpha=0.5, label="词频")
        # 叠加折线
        ax.plot(words, freqs, color="#fa8c16", linewidth=2)
        # 配置样式
        ax.set_title(f"新闻文本词频前20 - 面积图 {title_suffix}", fontsize=14, pad=20)
        ax.set_xlabel("词汇", fontsize=12)
        ax.set_ylabel("出现次数", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()  # 适配布局
        st.pyplot(fig)
    
    # 6. 热力图（词频矩阵，matplotlib）
    elif chart_type == "热力图（词频矩阵）":
        # 构造4行5列矩阵（前20词频）
        if len(freqs) < 20:
            # 不足20个时补0
            freqs += [0] * (20 - len(freqs))
            words += [""] * (20 - len(words))
        heatmap_data = np.array(freqs).reshape(4, 5)
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data, cmap="YlOrRd")
        # 设置坐标轴标签
        ax.set_xticks(range(5))
        ax.set_yticks(range(4))
        ax.set_xticklabels(words[:5], rotation=45)
        ax.set_yticklabels([f"第{i*5+1}-{i*5+5}名" for i in range(4)])
        # 标注数值
        for i in range(4):
            for j in range(5):
                text = ax.text(j, i, heatmap_data[i, j], ha="center", va="center", color="black", fontsize=10)
        # 配置样式
        plt.colorbar(im, ax=ax, label="词频")
        ax.set_title(f"新闻文本词频前20 - 热力图 {title_suffix}", fontsize=14, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
    
    # 7. 词云图（pyecharts）
    elif chart_type == "词云图（重点词可视化）":
        word_cloud = (
            WordCloud(init_opts=opts.InitOpts(width="1000px", height="600px"))
            .add(
                series_name="词频",
                data_pair=list(top20_words.items()),
                word_size_range=[15, 100],
                shape="circle"  # 词云形状：circle/rect/triangle等
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"新闻文本词频 - 词云图 {title_suffix}", title_textstyle_opts=opts.TextStyleOpts(font_size=16)),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )
        st_pyecharts(word_cloud)
    
    # 8. 美化表格（数据展示）
    elif chart_type == "表格（词频前20数据）":
        df = pd.DataFrame({
            "排名": range(1, len(words)+1),
            "词汇": words,
            "词频": freqs
        })
        # 表格美化（词频列渐变着色）
        styled_df = df.style.background_gradient(cmap="YlOrRd", subset=["词频"]) \
                          .set_properties(**{"text-align": "center"}) \
                          .set_table_styles([{"selector": "th", "props": [("font-size", "12px")]}])
        st.dataframe(styled_df, use_container_width=True)

# ====================== Streamlit主交互逻辑 ======================
if __name__ == "__main__":
    # 页面基础配置
    st.set_page_config(
        page_title="新闻文本词频分析系统",
        page_icon="📈",
        layout="wide"  # 宽屏布局
    )

    # 页面标题
    st.title("📈 新闻文本词频分析系统")
    st.divider()  # 分隔线

    # ---------------------- 侧边栏交互区 ----------------------
    with st.sidebar:
        st.title("🔧 交互配置")
        st.divider()
        # 1. 输入文章URL（多链接用英文逗号分割）
        url_input = st.text_area(
            label="📝 文章链接",
            placeholder="示例：https://xxx.com/1.html,https://xxx.com/2.html",
            height=100
        )
        # 2. 低频词过滤滑块
        min_freq = st.slider(
            label="🧹 最小词频阈值",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="过滤出现次数少于该值的词汇"
        )
        # 新增：分析模式选择
        analysis_mode = st.radio(
            label="🔍 分析模式",
            options=["单独分析每个链接", "合并所有链接分析"],
            index=0,
            help="选择「单独分析」将为每个链接输出独立结果；「合并分析」输出综合结果（原有逻辑）"
        )
        # 3. 图表类型选择
        chart_type = st.selectbox(
            label="📊 可视化图表",
            options=[
                "柱状图（词频前20）",
                "横向柱状图（词频前20）",
                "折线图（词频趋势）",
                "饼图（词频占比）",
                "面积图（词频累积）",
                "热力图（词频矩阵）",
                "词云图（重点词可视化）",
                "表格（词频前20数据）"
            ],
            index=0
        )
        st.divider()
        # 4. 执行按钮
        run_analysis = st.button("🚀 开始爬取并分析", type="primary")

    # ---------------------- 核心业务逻辑 ----------------------
    if run_analysis:
        if not url_input:
            st.error("❌ 请输入至少一个文章链接！")
        else:
            # 分割URL并去重/去空格
            ARTICLE_URLS = [url.strip() for url in url_input.split(",") if url.strip()]
            if not ARTICLE_URLS:
                st.error("❌ 链接格式错误！请用英文逗号分割多个链接")
            else:
                # 爬取并保存每篇文章
                st.subheader("🔍 爬取进度")
                file_list = []  # 存储爬取成功的文件路径
                for idx, url in enumerate(ARTICLE_URLS, start=1):
                    with st.expander(f"第{idx}篇：{url}", expanded=False):
                        st.info(f"正在爬取...")
                        content = get_webpage_content(url)
                        file_name = f"news{idx}.txt"
                        save_content_to_file(content, file_name)
                        file_list.append(file_name)  # 加入文件列表

                if not file_list:
                    st.error("❌ 未找到有效新闻文件（news1.txt/news2.txt等）！")
                else:
                    st.success(f"✅ 共找到{len(file_list)}个新闻文件，开始词频分析...")
                    st.divider()

                    # 模式1：单独分析每个链接
                    if analysis_mode == "单独分析每个链接":
                        for idx, file_name in enumerate(file_list, start=1):
                            st.subheader(f"📋 第{idx}个链接 - 词频排名前20（文件：{file_name}）")
                            # 单个文件词频分析
                            top20_words = get_single_file_word_freq(file_name, min_freq)
                            if top20_words:
                                df_top20 = pd.DataFrame({
                                    "排名": range(1, len(top20_words)+1),
                                    "词汇": list(top20_words.keys()),
                                    "词频": list(top20_words.values())
                                })
                                st.dataframe(df_top20, use_container_width=True)
                            else:
                                st.warning(f"⚠️ 第{idx}个链接无符合条件的词汇（请降低最小词频阈值）")
                            
                            # 单个文件可视化
                            st.subheader(f"📊 第{idx}个链接 - {chart_type}")
                            render_chart(chart_type, top20_words, title_suffix=f"（第{idx}个链接）")
                            st.divider()  # 分隔不同链接的结果

                    # 模式2：合并所有链接分析（保留原有逻辑）
                    else:
                        st.subheader("📋 所有链接合并 - 词频排名前20")
                        top20_words = get_merged_file_word_freq(file_list, min_freq)
                        if top20_words:
                            df_top20 = pd.DataFrame({
                                "排名": range(1, len(top20_words)+1),
                                "词汇": list(top20_words.keys()),
                                "词频": list(top20_words.values())
                            })
                            st.dataframe(df_top20, use_container_width=True)
                        else:
                            st.warning("⚠️ 无符合条件的词汇（请降低最小词频阈值）")

                        # 合并结果可视化
                        st.subheader(f"📊 所有链接合并 - {chart_type}")
                        render_chart(chart_type, top20_words, title_suffix="（所有链接合并）")

    # ---------------------- 辅助说明 ----------------------
    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        ### 使用步骤：
        1. 在侧边栏输入文章链接（多个链接用英文逗号`,`分割）；
        2. 调整「最小词频阈值」（过滤低频无意义词汇）；
        3. 选择「分析模式」：单独分析每个链接 / 合并所有链接分析；
        4. 选择需要展示的可视化图表类型；
        5. 点击「开始爬取并分析」按钮，等待结果。

        ### 适配说明：
        - 爬取逻辑默认适配class为`fulong_news_content`的网站，可修改`app.py`中`get_webpage_content`函数的class名适配其他网站；
        - 停用词表可在`stopwords.txt`中扩展/修改；
        - 生成的`news1.txt/news2.txt`等文件会保存在脚本同级目录。
        """)
