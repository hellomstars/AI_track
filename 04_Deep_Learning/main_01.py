# 4-1. 퍼셉트론

## 퍼셉트론 작동 예시 구현하기

# 학습 여부를 예측하는 퍼셉트론 함수
def Perceptron(x_1,x_2):
    
    # 설정한 가중치값을 적용
    w_0 = -5 
    w_1 = -1
    w_2 = 5
    
    # 활성화 함수에 들어갈 값을 계산
    output = w_0+w_1*x_1+w_2*x_2
    
    # 활성화 함수 결과를 계산
    if output < 0:
        y = 0
    else:
        y = 1
    
    return y, output

"""
1. perceptron의 예측 결과가 학습한다:1 이 나오도록
   x_1, x_2에 적절한 값을 입력하세요.
"""
x_1 = 0
x_2 = 2

result, go_out = Perceptron(x_1,x_2)

print("\n신호의 총합 : %d" % result) # output값 = 신호의 총합

if go_out > 0:
    print("학습 여부 : %d\n ==> 학습한다!" % go_out)
else:
    print("학습 여부 : %d\n ==> 학습하지 않는다!" % go_out)



## DIY 퍼셉트론 만들기
'''
1. 신호의 총합과 그에 따른 결과 0 또는 1을
   반환하는 함수 perceptron을 완성합니다.
   
   Step01. 입력 받은 값을 이용하여
           신호의 총합을 구합니다.
           
   Step02. 신호의 총합이 0 이상이면 1을, 
           그렇지 않으면 0을 반환하는 활성화 
           함수를 작성합니다.
'''
def perceptron(w, x):
    
    output = w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] *x[3] + w[0]
    
    if output >= 0:
        y = 1
    else:
        y = 0
    
    return y, output

# x_1, x_2, x_3, x_4의 값을 순서대로 list 형태로 저장
x = [1,2,3,4]

# w_0, w_1, w_2, w_3, w_4의 값을 순서대로 list 형태로 저장
w = [2, -1, 1, 3, -2]

# 퍼셉트론의 결과를 출력
y, output = perceptron(w,x)

print('output: ', output)
print('y: ', y)



## 퍼셉트론의 알맞은 가중치 찾기
def perceptron(w, x):
    output = w[1] * x[0] + w[2] * x[1] + w[0]
    if output >= 0:
        y = 1
    else:
        y = 0
    return y

# Input 데이터
X = [[0,0], [0,1], [1,0], [1,1]]

'''
1. perceptron 함수의 입력으로 들어갈 가중치 값을 입력해주세요.
   순서대로 w_0, w_1, w_2에 해당됩니다.
'''
w = [-2, 1, 1]

# AND Gate를 만족하는지 출력하여 확인
print('perceptron 출력')

for x in X:
    print('Input: ',x[0], x[1], ', Output: ',perceptron(w, x))