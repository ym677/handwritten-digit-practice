from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# 使用 app.py 所在目录定位模型，不受 PyCharm 工作目录的影响。
model_path = Path(__file__).resolve().parent / "digits_model.joblib"
model = joblib.load(model_path)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return {"message": "请先选择图片"}, 400

    try:
        # 和练习脚本保持一致：灰度、8×8、亮度反转、缩放至 0～16。
        with Image.open(uploaded_file.stream) as image:
            small_image = image.convert("L").resize((8, 8))
            pixels = np.asarray(small_image, dtype=float)
    except (UnidentifiedImageError, OSError, ValueError):
        return {"message": "无法读取图片，请选择有效的图片文件"}, 400

    processed_pixels = (255 - pixels) / 255 * 16
    model_input = processed_pixels.reshape(1, 64)
    prediction = int(model.predict(model_input)[0])
    return {"filename": uploaded_file.filename, "prediction": prediction}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
