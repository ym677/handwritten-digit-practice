from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 打开图片
image = Image.open("手绘图片_6.png")

# 2. 转为灰度图，再缩小到 8×8
image_grey = image.convert("L")
trans_image = image_grey.resize((8, 8))

# 3. 把图片转成 NumPy 数组
pixels = np.asarray(trans_image, dtype=float)

# 4. 反转亮度，并把像素范围从 0～255 缩放到 0～16
processed_pixels = (255 - pixels) / 255 * 16

# 5. 检查结果
print("图片尺寸：", trans_image.size)
print("图片模式：", trans_image.mode)
print("数组形状：", processed_pixels.shape)
print("处理后的像素值：")
print(processed_pixels)

# 6. 显示处理后的图片
plt.imshow(processed_pixels, cmap="gray", interpolation="nearest",
           vmin=0, vmax=16)
plt.title("Processed digit 6")
plt.show()

model_input = processed_pixels.reshape(1, 64)
print(model_input.shape)


import joblib

model = joblib.load('digits_model.joblib')
res = model.predict(model_input)
print(res)

from sklearn.datasets import load_digits

digits = load_digits()

# 从数据集中找一张标签为 6 的图片
example_six = digits.images[digits.target == 6][0]

# 左右各显示一张图片
fig, axes = plt.subplots(1, 2)

axes[0].imshow(processed_pixels, cmap="gray", vmin=0, vmax=16)
axes[0].set_title("My digit")

axes[1].imshow(example_six, cmap="gray", vmin=0, vmax=16)
axes[1].set_title("Dataset digit 6")

plt.show()