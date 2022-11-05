# 2-2. 데이터 핸들링을 위한 라이브러리 NumPy

## 배열 만들기
import numpy as np

array = np.array(range(5)) # 0~4의 시퀀스를 만들어봅시다.
print(array) # 출력 [0 1 2 3 4]



## 배열의 기초(1)
import numpy as np

print("1차원 array")
array = np.array(range(10))
print(array) #출력 : [0 1 2 3 4 5 6 7 8 9]

# 1. type()을 이용하여 array의 자료형을 출력해보세요.
print(type(array)) #출력 : <class 'numpy.ndarray'>

# 2. ndim을 이용하여 array의 차원을 출력해보세요.
print(array.ndim) #출력 : 1

# 3. shape을 이용하여 array의 모양을 출력해보세요.
print(array.shape) #출력 : (10,)

# 4. size를 이용하여 array의 크기를 출력해보세요.
print(array.size) #출력 : 10

# 5. dtype을 이용하여 array의 dtype(data type)을 출력해보세요.
print(array.dtype) #출력 : int64

# 6. array의 5번째 요소를 출력해보세요. (array[5])
print(array[5]) #출력 : 5

# 7. array의 3번째 요소부터 5번째 요소까지 출력해보세요. (array[3:6])
print(array[3:6]) #출력 : [3 4 5]



## 배열의 기초(2)
import numpy as np


print("2차원 array")
matrix = np.array(range(1,16))  #1부터 15까지 들어있는 (3,5)짜리 배열을 만듭니다.
matrix.shape = 3,5
print(matrix) #출력 : [[1 2 3 4 5][6 7 8 9 10][11 12 13 14 15]]


# 1. type을 이용하여 matrix의 자료형을 출력해보세요.
print(type(matrix)) #출력 : <class 'numpy.ndarray'>

# 2. ndim을 이용하여 matrix의 차원을 출력해보세요.
print(matrix.ndim) #출력 : 2

# 3. shape을 이용하여 matrix의 모양을 출력해보세요.
print(matrix.shape) #출력 : (3, 5)

# 4. size를 이용하여 matrix의 크기를 출력해보세요.
print(matrix.size) #출력 : 15

# 5. dtype을 이용하여 matrix의 dtype(data type)을 출력해보세요.
print(matrix.dtype) #출력 : int64

# 6. astype을 이용하여 matrix의 dtype을 str로 변경하여 출력해보세요.
print(matrix.astype('str')) #출력 : [['1' '2' '3' '4' '5']['6' '7' '8' '9' '10']['11' '12' '13' '14' '15']]

# 7. matrix의 (2,3) 인덱스의 요소를 출력해보세요.
print(matrix[2,3]) #출력 : 14

# 8. matrix의 행은 인덱스 0부터 인덱스 1까지 (0:2), 열은 인덱스 1부터 인덱스 3까지 (1:4) 출력해보세요.
print(matrix[0:2, 1:4]) #출력 : [[2 3 4][7 8 9]]



## Indexing & Slicing
import numpy as np

matrix = np.arange(1, 13, 1).reshape(3, 4)
print(matrix)
# matrix는 아래와 같습니다.
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# 1. Indexing을 통해 값 2를 출력해보세요.
answer1 = matrix[0, 1] # 2는 0행 1열에 있습니다.

# 2. Slicing을 통해 매트릭스 일부인 9, 10을 가져와 출력해보세요.
answer2 = matrix[2:, :2] # 2행 첫 두 개 열에 9, 10 이 있습니다. (2:, :2)

# 3. Boolean indexing을 통해 5보다 작은 수를 찾아 출력해보세요.
answer3 = matrix[matrix < 5]

# 4. Fancy indexing을 통해 두 번째 행만 추출하여 출력해보세요.
answer4 = matrix[ [ 1 ] ] # 두번째 행의 인덱스는 1입니다.

# 위에서 구한 정답을 출력해봅시다.
print(answer1) #출력 : [2]
print(answer2) #출력 : [[9 10]]
print(answer3) #출력 : [[1 2 3 4]]
print(answer4) #출력 : [[5 6 7 8]]