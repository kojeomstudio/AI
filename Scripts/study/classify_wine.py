import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # 데이터셋 분리에 사용.

from sklearn.linear_model import LinearRegression # 로지스틱 회귀 ( 회귀라는 이름이지만 실제로는 분류에 사용 -> 다항식 )
from sklearn.linear_model import Ridge # 특성에 대한 규제를 하는 릿지
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler # 표준화 (데이터 전처리))
import matplotlib.pyplot as plt

# CSV 파일 불러오기
file_path = '../Datas/WineQT.csv'
wine_data = pd.read_csv(file_path)

# 데이터 확인
print("==== wine_data.head() ====")
print("{0}".format(wine_data.head()))

# 데이터 정보 확인
print("wine_data.info() : {0}".format(wine_data.info()))

# 기초 통계량 확인
print("wine_data.describe()")
print("{0}".format(wine_data.describe()))

# 특성과 레이블 분리 (예시로 'quality'를 레이블로 사용)
# 여러 특성을 이용해서 와인의 '품질'을 예측한다. 라는 문제로 생각해본다.
# 
X = wine_data.drop('quality', axis=1) # quality 컬럼을 제외한 나머지는 feature (=학습에 이용)
y = wine_data['quality'] # quality는 라벨링 (=정답)

# 데이터 나누기
x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 표준화
#scaler = StandardScaler()
#x_train_scaled = scaler.fit_transform(x_train)
#x_test_scaled = scaler.transform(x_test)

# 나눈 데이터의 크기 출력
print(f'Training set: {x_train.shape}, {y_train.shape}')
print(f'Validation set: {x_val.shape}, {y_val.shape}')
print(f'Test set: {x_test.shape}, {y_test.shape}')

lr = LinearRegression()
lr.fit(x_train, y_train)

print("linearRegression -> coef : {0}".format(lr.coef_))

print("linearRegression -> score : {0}".format(lr.score(x_test, y_test)))
#print("linearRegression -> predict : {0}".format(lr.predict(x_test)))

# SVM 사용
# SVM 모델 초기화 및 학습
svm_model = SVC(kernel='linear')  # 커널 함수로 선형 커널 사용
svm_model.fit(x_train, y_train)

print("svm_model --> score : {0}".format(svm_model.score(x_train, y_train)))

# 릿지 회귀 모델 초기화
#ridge_model = Ridge(alpha=1.0)

# 릿지 모델 학습
#ridge_model.fit(x_train, y_train)
#predict_rm = ridge_model.predict(x_test, y_test)
#print("ridge_model -> score : {0}".format(ridge_model.score(x_test, y_test)))