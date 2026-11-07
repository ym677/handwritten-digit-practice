from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib


# 1. 加载数据
digits = load_digits()

x = digits.data
y = digits.target

random_state = 42


# 2. 先划出测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=random_state,
    stratify=y
)

print("训练和验证部分：", X_trainval.shape, y_trainval.shape)
print("测试集：", X_test.shape, y_test.shape)


# 3. 再从训练部分划出验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=random_state,
    stratify=y_trainval
)

print("训练集：", X_train.shape, y_train.shape)
print("验证集：", X_val.shape, y_val.shape)


# 4. 比较不同的 k
maxacc = -1.0
maxi = None

for k in range(1, 6):
    Knn_model = KNeighborsClassifier(n_neighbors=k)
    Knn_model.fit(X_train, y_train)


    accuracy = Knn_model.score(X_val, y_val)


    print("k =", k, "验证集准确率：", accuracy)

    # 如果当前结果更好，就记录当前的准确率和 k
    if accuracy > maxacc:
        maxacc = accuracy
        maxi = k


print("选出的 k：", maxi)
print("最高验证集准确率：", maxacc)


# 5. 重新创建并训练选出的模型
Knn_model = KNeighborsClassifier(n_neighbors=maxi)
Knn_model.fit(X_train, y_train)

# 用选出的模型预测验证集
x_predict = Knn_model.predict(X_val)


# 6. 找出预测错误的样本
wrong_count = 0

for index in range(len(y_val)):
    if x_predict[index] != y_val[index]:
        print(
            "验证集索引：", index,
            "正确数字：", y_val[index],
            "预测数字：", x_predict[index]
        )
        wrong_count += 1
        image = X_val[index].reshape(8,8)
        plt.imshow(image, cmap="gray",interpolation="nearest")
        plt.title(f"True:{y_val[index]},Predicted:{x_predict[index]}")
        plt.show()


print("验证集预测错误数量：", wrong_count)

test_accuracy = Knn_model.score(X_test, y_test)
print(test_accuracy)
joblib.dump(Knn_model, 'digits_model.joblib')