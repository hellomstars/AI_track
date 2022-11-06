# 3-1. 자료 형태의 이해

## 수치형 자료, 범주형 자료 만들어 보기
# 수치형 자료 중 이산형 자료를 만들어 봅니다.
ages = [10, 32, 48, 71, 50]

# 수치형 자료 중 연속형 자료를 만들어 봅니다.
weights = [42.3, 88.23, 51.0, 72.35, 58.8]

# 범주형 자료 중 명목형 자료를 만들어 봅니다.
blood_types = ['A', 'O' ,'A', 'AB', 'B', 'O']

# 범주형 자료 중 순위형 자료를 만들어 봅니다.
grades = ['A', 'B', 'C', 'D', 'F', 'A']



## 범주형 자료의 요약 - 도수분포표
import pandas as pd 

# drink 데이터
drink = pd.read_csv("drink.csv")

#도수 계산
drink_freq = drink[drink["Attend"] == 1]["Name"].value_counts()

print("도수분포표")
print(drink_freq)



## 범주형 자료의 요약 - 막대 그래프
import matplotlib.pyplot as plt
   
# 술자리 참석 상대도수 데이터 
labels = ['A', 'B', 'C', 'D', 'E']
ratio = [4,3,2,2,1]
    
#막대 그래프
fig, ax = plt.subplots()

"""
1. 막대 그래프를 만드는 코드를 작성해 주세요
"""
plt.bar(labels, ratio)

# 출력에 필요한 코드
plt.show()
fig.savefig("bar_plot.png")



## 수치형 자료의 요약 - 평균
import numpy as np

coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 평균계산
"""
cf_mean = coffee.mean() # 또는 np.mean(coffee)

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Mean :", round(cf_mean,2))



## 수치형 자료의 요약 - 표준편차
from statistics import stdev
import numpy as np

coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 표준편차 계산
"""
cf_std = stdev(coffee)

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Sample std.Dev : ", round(cf_std,2))



## 수치형 자료의 요약 -히스토그램
import numpy as np
import matplotlib.pyplot as plt

# 카페인 데이터
coffee = np.array([202,177,121,148,89,121,137,158])

fig, ax = plt.subplots()

"""
1. 히스토그램을 그리는 코드를 작성해 주세요
"""
plt.hist(coffee)

# 히스토그램을 출력합니다.
plt.show()
fig.savefig("hist_plot.png")