# 3-2. 데이터 전 처리하기

## 명목형 자료 변환하기 - 수치 맵핑
import pandas as pd

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Sex'].head())
"""
1. replace를 사용하여 male -> 0, female -> 1로 변환합니다.
"""
titanic = titanic.replace({'male': 0, 'female': 1})

# 변환한 성별 데이터를 출력합니다.
print('\n변환 후: \n',titanic['Sex'].head())



## 명목형 자료 변환하기 - 더미 방식
import pandas as pd

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Embarked'].head())
"""
1. get_dummies를 사용하여 변환합니다.
"""
dummies = pd.get_dummies(titanic[['Embarked']])

# 변환한 Embarked 데이터를 출력합니다.
print('\n변환 후: \n',dummies.head())



## 수치형 자료 변환하기 - 정규화
import pandas as pd

"""
1. 정규화를 수행하는 함수를 구현합니다.
"""
def normal(data):
    data = (data-data.min())/(data.max()-data.min())
    return data

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

# normal 함수를 사용하여 정규화합니다.
Fare = normal(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())



## 수치형 자료 변환하기 - 표준화
import pandas as pd

"""
1. 표준화를 수행하는 함수를 구현합니다.
"""
def standard(data):
    data = (data-data.mean())/data.std()
    return data
    
# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

# standard 함수를 사용하여 표준화합니다.
Fare = standard(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())



## 결측값 처리하기
import pandas as pd

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
# 변수 별 데이터 수를 확인하여 결측 값이 어디에 많은지 확인합니다.
print(titanic.info(),'\n')

"""
1. Cabin 변수를 제거합니다.
"""
titanic_1 = titanic.drop(columns=['Cabin'])

# Cabin 변수를 제거 후 결측값이 어디에 남아 있는지 확인합니다.
print('Cabin 변수 제거')
print(titanic_1.info(),'\n')

"""
2. 결측값이 존재하는 샘플 제거합니다.
"""
titanic_2 = titanic_1.dropna()

# 결측값이 존재하는지 확인합니다.
print('결측값이 존재하는 샘플 제거')
print(titanic_2.info())



## 이상치 처리하기
import pandas as pd
import numpy as np

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# Cabin 변수를 제거합니다.
titanic_1 = titanic.drop(columns=['Cabin'])

# 결측값이 존재하는 샘플 제거합니다.
titanic_2 = titanic_1.dropna()

# (Age 값 - 내림 Age 값) 0 보다 크다면 소수점을 갖는 데이터로 분류합니다.
outlier = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) > 0 ]['Age']

print('소수점을 갖는 Age 변수 이상치')
print(outlier)
print('이상치 처리 전 샘플 개수: %d' %(len(titanic_2)))
print('이상치 개수: %d' %(len(outlier)))

"""
1. 이상치를 처리합니다
 -> np.floor() 를 사용하면 입력값의 소수점을 제외한 정수 부분을 얻을 수 있습니다.
 -> titanic_2['Age'] 에서 np.floor(titanic_2['Age']) 을 뺀 값이 0이면 정수입니다.
 -> 즉, titanic_2 에서 titanic_2['Age']-np.floor(titanic_2['Age']) 값이 0 인 샘플만 titanic_3에 저장합니다.
"""
titanic_3 = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) == 0 ]
print('이상치 처리 후 샘플 개수: %d' %(len(titanic_3)))



## 데이터 분리하기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# Cabin 변수를 제거합니다.
titanic_1 = titanic.drop(columns=['Cabin'])

# 결측값이 존재하는 샘플 제거합니다.
titanic_2 = titanic_1.dropna()

# 이상치를 처리합니다.
titanic_3 = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) == 0 ]
print('전체 샘플 데이터 개수: %d' %(len(titanic_3)))

"""
1. feature 데이터와 label 데이터를 분리합니다.
"""
X = titanic_3.drop(columns=['Survived']) # Survived 변수를 제거하여 X에 저장합니다.
y = titanic_3['Survived'] # Survived 변수를 y에 저장합니다.
print('X 데이터 개수: %d' %(len(X)))
print('y 데이터 개수: %d' %(len(y)))

"""
2. X,y 데이터를 학습용, 평가용 데이터로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(feature 데이터, label 데이터, test_size = 0~1 값, random_state = 랜덤시드값)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 분리한 데이터의 개수를 출력합니다.
print('학습용 데이터 개수: %d' %(len(X_train)))
print('평가용 데이터 개수: %d' %(len(X_test)))