# 3-4. 지도학습 - 분류

## sklearn을 사용한 의사결정나무 - 데이터 전 처리
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# sklearn에 저장된 데이터를 불러 옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']

"""
1. 학습용 평가용 데이터로 분리합니다
"""
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# 원본 데이터 출력
print('원본 데이터 : \n',df.head(),'\n')

# 전 처리한 데이터 5개만 출력합니다
print('train_X : ')
print(train_X[:5],'\n')
print('train_Y : ')
print(train_Y[:5],'\n')

print('test_X : ')
print(test_X[:5],'\n')
print('test_Y : ')
print(test_Y[:5])



## sklearn을 사용한 의사결정나무 - 학습하기
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

# sklearn에 저장된 데이터를 불러 옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])
Y = df['클래스']
    
# 학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

"""
1. DTmodel에 의사결정나무 모델을 초기화 하고 학습합니다.
"""
DTmodel = DecisionTreeClassifier()
DTmodel.fit(train_X, train_Y)

# 학습한 결과를 출력합니다
plt.rc('font', family='NanumBarunGothic')
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DTmodel, 
                   feature_names=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'],  
                   class_names=['setosa', 'versicolor', 'virginica'],
                   filled=True)

fig.savefig("decision_tree.png")



## sklearn을 사용한 의사결정나무 - 예측하기
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# sklearn에 저장된 데이터를 불러 옵니다.
X, Y = load_iris(return_X_y = True)

# DataFrame으로 변환
df = pd.DataFrame(X, columns=['꽃받침 길이','꽃받침 넓이', '꽃잎 길이', '꽃잎 넓이'])
df['클래스'] = Y

X = df.drop(columns=['클래스'])

Y = df['클래스']
    
# 학습용 평가용 데이터로 분리합니다
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state = 42)

# DTmodel에 의사결정나무 모델을 초기화 하고 학습합니다
DTmodel = DecisionTreeClassifier()
DTmodel.fit(train_X, train_Y)

"""
1. test_X에 대해서 예측합니다.
"""
pred_X = DTmodel.predict(test_X)
print('test_X에 대한 예측값 : \n{}'.format(pred_X))



## 간단한 의사결정나무 만들기
import pandas as pd

# 풍속을 threshold 값에 따라 분리하는 의사결정나무 모델 함수
def binary_tree(data, threshold):
    
    yes = []
    no = []
    
    # data로부터 풍속 값마다 비교를 하기 위한 반복문
    for wind in data['풍속']:
    
        # threshold 값과 비교하여 분리합니다.
        if wind > threshold:
            yes.append(wind)
        else:
            no.append(wind)
    
    # 예측한 결과를 DataFrame 형태로 저장합니다.
    data_yes = pd.DataFrame({'풍속': yes, '예상 지연 여부': ['Yes']*len(yes)})
    data_no = pd.DataFrame({'풍속': no, '예상 지연 여부': ['No']*len(no)})
    
    return data_no.append(data_yes,ignore_index=True)

# 풍속에 따른 항공 지연 여부 데이터
Wind = [1, 1.5, 2.5, 5, 5.5, 6.5]
Delay  = ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']

# 위 데이터를 DataFrame 형태로 저장합니다.
data = pd.DataFrame({'풍속': Wind, '지연 여부': Delay})
print(data,'\n')

"""
1. binary_tree 모델을 사용하여 항공 지연 여부를 예측합니다.
"""
data_pred = binary_tree(data, threshold = 3)
print(data_pred,'\n')



## 
