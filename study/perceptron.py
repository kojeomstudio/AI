import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris_data = load_iris(as_frame=True)  # iris 데이터를 pandas DataFrame으로 로드
iris_df = iris_data.frame  # DataFrame 객체를 추출

x = iris_df[["petal length (cm)", "petal width (cm)"]].values  # 'petal length'와 'petal width'를 선택
y = (iris_df["target"] == 0)  # target 컬럼에서 'Setosa' 종(0)을 선택

per_clf = Perceptron(random_state=42)
per_clf.fit(x, y)

print(f"per_clf score : {per_clf.score(x, y)}")  # x와 y를 입력으로 스코어 계산

x_new = [[2, 0.5], [3, 1]]
print(f"predict by x_new data : {per_clf.predict(x_new)}")
