import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # 데이터셋 분리에 사용.
from sklearn.linear_model import SGDClassifier # SGD분류
from sklearn.model_selection import cross_val_score # 교차 검증

import seaborn as sns

from sklearn.preprocessing import StandardScaler # 표준화 (데이터 전처리))
import matplotlib.pyplot as plt

# CSV 파일 불러오기
file_path = '../Datas/heart.csv'
heart_data = pd.read_csv(file_path)

# 데이터 확인
print("==== heart_data.head() ====")
print("{0}".format(heart_data.head()))

# 데이터 정보 확인
print("heart_data.info() : {0}".format(heart_data.info()))

# 기초 통계량 확인
print("heart_data.describe")
print("{0}".format(heart_data.describe()))

# 히스토그램 그리기
#heart_data['age'].hist()
#plt.show()

# 상관관계 행렬 및 히트맵
#corr_matrix = heart_data.corr()
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#plt.show()

# 페어플롯 (특히 다변량 데이터에 유용)
#sns.pairplot(heart_data)
#plt.show()

x = heart_data.drop('output', axis=1) #  (=학습에 이용)
y = heart_data['output'] # (=정답)

print("data features : {0}".format(x.columns))

# 데이터 나누기
# (훈련, 테스트, 검증)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 모델 초기화
# 규제를 좀 더 강하게 세팅.
sgd_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42, penalty='elasticnet', alpha=0.1)

# 모델 학습
sgd_clf.fit(x_train, y_train)

print("sgd_clf.classes_ : {0}".format(sgd_clf.classes_))


# 예측
y_pred = sgd_clf.predict(x_test)
print("prediction y : {0}".format(y_pred))

# 성능 평가 
print(f"score_value (from test data) : {sgd_clf.score(x_test_scaled, y_test):.2f}")
print(f"score_value (from train data) : {sgd_clf.score(x_train_scaled, y_train):.2f}")

# 교차 검증으로 성능 평가.
scores = cross_val_score(sgd_clf, x_train_scaled, y_train, cv=5)
print(f"cross vali score: {scores}")
print(f"cross vali score mean: {scores.mean()}")