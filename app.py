from flask import Flask,render_template
#导入创建网页的flask和渲染用的render

app = Flask(__name__)


@app.get("/")
#进入app首页的时候直接调用index函数渲染
def index():
    return render_template("index.html")


if __name__ == "__main__":
    #启动网页，主机号、端口号、关闭调试模式
    app.run(host="127.0.0.1",port=5000,debug=False)