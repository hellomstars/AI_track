# 3-3. 지도학습 - 회귀

## 단순 선형 회귀 분석하기 - 데이터 전 처리
import pandas as pd
from sklearn.linear_model import LinearRegression
'''
sklearn의 LinearRegression 입력 값 형태

LinearRegression 모델의 입력값으로는 Pandas의 DataFrame의 feature (X) 데이터와 Series 형태의 label (Y) 데이터를 입력 받을 수 있습니다.
X, Y의 샘플의 개수는 같아야 합니다.
'''

X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

"""
1. X의 형태를 변환하여 train_X에 저장합니다.
"""
train_X = pd.DataFrame(X, columns=['X']) # X를 column 명이 'X'인 Dataframe으로 변환합니다.

"""
2. Y의 형태를 변환하여 train_Y에 저장합니다.
"""
train_Y = pd.Series(Y) # Y를 Series로 변환합니다.

# 변환된 데이터를 출력합니다.
print('전 처리한 X 데이터: \n {}'.format(train_X))
print('전 처리한 X 데이터 shape: {}\n'.format(train_X.shape))

print('전 처리한 Y 데이터: \n {}'.format(train_Y))
print('전 처리한 Y 데이터 shape: {}'.format(train_Y.shape))



## 단순 선형 회귀 분석하기 - 학습하기
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

"""
1. 모델을 초기화 합니다.
"""
lrmodel = LinearRegression() # LinearRegression 모델을 초기화 합니다.

"""
2. train_X, train_Y 데이터를 학습합니다.
"""
lrmodel.fit(train_X, train_Y) # train_X와 train_Y를 이용하여 모델을 학습 시킵니다.


# 학습한 결과를 시각화하는 코드입니다.
plt.scatter(X, Y) 
plt.plot([0, 10], [lrmodel.intercept_, 10 * lrmodel.coef_[0] + lrmodel.intercept_], c='r') #y절편값 = .intercept_ , x값 10에 대한 기울기
plt.xlim(0, 10) 
plt.ylim(0, 10) 
plt.title('Training Result')
plt.savefig("test.png") 



## 단순 선형 회귀 분석하기 - 예측하기
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

X = [8.70153760, 3.90825773, 1.89362433, 3.28730045, 7.39333004, 2.98984649, 2.25757240, 9.84450732, 9.94589513, 5.48321616]
Y = [5.64413093, 3.75876583, 3.87233310, 4.40990425, 6.43845020, 4.02827829, 2.26105955, 7.15768995, 6.29097441, 5.19692852]

train_X = pd.DataFrame(X, columns=['X'])
train_Y = pd.Series(Y)

# 모델을 트레이닝합니다.
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

"""
1. train_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(train_X) # predict() 를 이용하여 예측합니다.
print('train_X에 대한 예측값 : \n{}\n'.format(pred_X))
print('실제값 : \n{}'.format(train_Y))



## 다중 회귀 분석하기 - 데이터 전 처리
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Advertising.csv")

print('원본 데이터 샘플 :')
print(df.head(),'\n')

# 입력 변수로 사용하지 않는 Unnamed: 0 변수 데이터를 삭제합니다
df = df.drop(columns=['Unnamed: 0'])

"""
1. Sales 변수는 label 데이터로 Y에 저장하고 나머진 X에 저장합니다.
"""
X = df.drop(columns=['Sales'])
Y = df['Sales']

"""
2. 2:8 비율로 (test_size = 0.2) X와 Y를 학습용과 평가용 데이터로 분리합니다.
"""
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42) 

# 전 처리한 데이터를 출력합니다
print('train_X : ')
print(train_X.head(),'\n')
print('train_Y : ')
print(train_Y.head(),'\n')

print('test_X : ')
print(test_X.head(),'\n')
print('test_Y : ')
print(test_Y.head())



## 다중 회귀 분석하기 - 학습하기
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

"""
1.  다중 선형 회귀 모델을 초기화 하고 학습합니다
"""
lrmodel = LinearRegression() # LinearRegression 모델을 초기화 합니다.
lrmodel.fit(train_X, train_Y) # train_X와 train_Y 데이터로 모델을 학습합니다.

"""
2. 학습된 파라미터 값을 불러옵니다
"""
beta_0 = lrmodel.intercept_ # y절편 (기본 판매량)
beta_1 = lrmodel.coef_[0] # 1번째 변수에 대한 계수 (페이스북)
beta_2 = lrmodel.coef_[1] # 2번째 변수에 대한 계수 (TV)
beta_3 = lrmodel.coef_[2] # 3번째 변수에 대한 계수 (신문)

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)



## 다중 회귀 분석하기 - 예측하기
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

print('test_X : ')
print(test_X)

"""
1. test_X에 대해서 예측합니다.
"""
pred_X = lrmodel.predict(test_X) # predict()를 활용해서 예측합니다.
print('test_X에 대한 예측값 : \n{}\n'.format(pred_X))

# 새로운 데이터 df1을 정의합니다
df1 = pd.DataFrame(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]), columns=['FB', 'TV', 'Newspaper'])
print('df1 : ')
print(df1)

"""
2. df1에 대해서 예측합니다.
"""
pred_df1 = lrmodel.predict(df1) # predict()를 활용해서 예측합니다.
print('df1에 대한 예측값 : \n{}'.format(pred_df1))



## 회귀 알고리즘 평가 지표 - RSS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 RSS 값을 계산합니다
"""
RSS_train = np.sum( (train_Y - pred_train) ** 2) # 예측값과 실제값의 오차제곱합을 구합니다.
print('RSS_train : %f' % RSS_train)

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 RSS 값을 계산합니다
"""
RSS_test = np.sum( (test_Y - pred_test) ** 2) # 예측값과 실제값의 오차제곱합을 구합니다.
print('RSS_test : %f' % RSS_test)



## 회귀 알고리즘 평가 지표 - MSE, MAE
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 MSE, MAE 값을 계산합니다
"""
MSE_train = mean_squared_error(train_Y, pred_train) # mean_squared_error() 를 활용해서 MSE를 계산합니다.
MAE_train = mean_absolute_error(train_Y, pred_train) # mean_absolute_error() 를 활용해서 MAE를 계산합니다.
print('MSE_train : %f' % MSE_train)
print('MAE_train : %f' % MAE_train)

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 MSE, MAE 값을 계산합니다
"""
MSE_test = mean_squared_error(test_Y, pred_test) 
MAE_test = mean_absolute_error(test_Y, pred_test)
print('MSE_test : %f' % MSE_test)
print('MAE_test : %f' % MAE_test)



## 회귀 알고리즘 평가 지표 - R2
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 데이터를 읽고 전 처리합니다
df = pd.read_csv("data/Advertising.csv")
df = df.drop(columns=['Unnamed: 0'])

X = df.drop(columns=['Sales'])
Y = df['Sales']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 다중 선형 회귀 모델을 초기화 하고 학습합니다
lrmodel = LinearRegression()
lrmodel.fit(train_X, train_Y)

# train_X 의 예측값을 계산합니다
pred_train = lrmodel.predict(train_X)

"""
1. train_X 의 R2 값을 계산합니다
"""
R2_train = r2_score(train_Y, pred_train) # r2_score()를 활용하여 R2값을 계산합니다.
print('R2_train : %f' % R2_train)

# test_X 의 예측값을 계산합니다
pred_test = lrmodel.predict(test_X)

"""
2. test_X 의 R2 값을 계산합니다
"""
R2_test = r2_score(test_Y, pred_test)
print('R2_test : %f' % R2_test)