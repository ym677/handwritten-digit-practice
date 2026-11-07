import joblib
from sklearn.datasets import load_digits


model = joblib.load('digits_model.joblib')

picture = load_digits()
predict_res = model.predict(picture.data[0:1])
print(predict_res,picture.target[0])