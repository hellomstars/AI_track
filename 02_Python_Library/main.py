#cal 함수 불러오기
import cal

var1 = cal.modelName # cal.modelName을 var1에 저장하세요.

var2 = cal.plus(3, 4) # cal.plus를 활용하여 3+4를 구해보세요.

var3 = cal.minus(7, 2) # cal.minus를 활용하여 7-2를 구해보세요.

print(var1, var2, var3)
# 출력 : ELI-C2 7 5

#모듈 불러오기
from random import randrange # from, import를 활용하여 random 모듈에서 randrange 함수를 불러오세요.
import math # math 모듈을 불러오세요.

var1 = randrange(1, 11)
var2 = math.log(5184, 72) # math.log 함수를 사용해보세요.

#웹페이지 구성 확인하기
from urllib.request import urlopen # url 패키지의 request 모듈에서 urlopen 함수를 불러오세요. 해당 함수는 url의 html파일 불러옴.

webpage = urlopen("https://en.wikipedia.org/wiki/Lorem_ipsum").read().decode("utf-8")

print(webpage)