from flask import Flask, render_template, request
from analysis.text_processor import text_segmentation, sentiment_analysis, extract_keywords
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)  # 集成Bootstrap美化页面

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 获取用户输入的文本
        text = request.form.get("text", "").strip()
        if not text:
            return render_template("index.html", error="请输入要分析的文本！")
        
        # 执行文本分析
        seg_result = text_segmentation(text)
        sentiment_result = sentiment_analysis(text)
        keywords_result = extract_keywords(text)
        
        # 渲染结果页
        return render_template(
            "result.html",
            original_text=text,
            seg_result=seg_result,
            sentiment=sentiment_result,
            keywords=keywords_result
        )
    # GET请求渲染首页
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
