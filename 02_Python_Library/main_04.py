# 2-4. Matplotlib 데이터 시각화

## 선 그래프
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#이미 입력되어 있는 코드의 다양한 속성값들을 변경해 봅시다.
x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(
    x, x, label='y=x',
    linestyle='-', # 실선으로 변경해보세요.
    marker='.', # 점으로 변경해보세요.
    color='blue'
)
ax.plot(
    x, x**2, label='y=x^2',
    linestyle='-.', # 대시점선으로 변경해보세요.
    marker=',', # 픽셀로 변경해보세요.
    color='red'
)
ax.set_xlabel("x")
ax.set_ylabel("y")


fig.savefig("plot.png")



## 히스토그램
fig, ax = plt.subplots() # 그래프 그리기 위한 fig와 ax설정
data = np.random.randn(1000) # 시각화하고자 하는 데이터 설정
ax.hist(data, bins=50) # 시각화 할 막대 수를 bins로 data를 가지고 히스토그램 그리기



## 데이터 범례
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(
    x, x, label='y=x',
    linestyle='-',
    marker='.',
    color='blue'
)
ax.plot(
    x, x**2, label='y=x^2',
    linestyle='-.',
    marker=',',
    color='red'
)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.legend(
    loc='center left', # 좌측 가운데데로 변경해보세요.
    shadow=True,
    fancybox=True,
    borderpad=2
)

fig.savefig("plot.png")



## 막대 그래프 & 히스토그램
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fname='./NanumBarunGothic.ttf' #matplotlib 의 pyplot으로 그래프를 그릴 때, 기본 폰트는 한글을 지원하지 않습니다. 따라서 한글을 지원하는 나눔바른고딕 폰트로 바꾼 코드입니다.
font = fm.FontProperties(fname = fname).get_name()
plt.rcParams["font.family"] = font

# Data set
x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"])
y = np.array([13, 10, 17, 8, 7])
z = np.random.randn(1000)

fig, axes = plt.subplots(1, 2, figsize=(8, 4)) # 1*2의 모양으로 그래프를 그리도록 합니다. 그래프를 2개 그리고, 가로로 배치한다는 의미.

# Bar 그래프
axes[0].bar(x, y)
# 히스토그램
axes[1].hist(z, bins = 200)

fig.savefig("plot.png")