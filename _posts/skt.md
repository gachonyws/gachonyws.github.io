1.
```
def solution(A, B, q):
    # write your code in Python 3.6
    specificity = 0
    sensitivity = 0
    total_neg = 0
    total_pos = 0
    fp=0
    tn=0
    fn=0
    tp=0

    for i in range(len(A)):
        if B[i] == 0: #진짜음성
            total_neg += 1
            if A[i] != 0: #음성인데 양성이라 예측해서 실패: FP
                fp += 1
            if A[i] == 0: #음성인데 음성이라 예측 성공:TN
                tn += 1
        if B[i] == 1: #진짜 양성
            total_pos += 1
            if A[i] == 0: #양성인데 음성이라 예측해서 실패 FN
                fn += 1
            if A[i] != 0: #양성인데 양성이라 예측 성공 TP
                tp += 1

    specificity = (tn+fp) / (total_neg+total_pos)
    sensitivity = (total_pos) / (total_neg+total_pos)


    if q == False:
        return sensitivity

    else :
        return specificity

```

2.
```
A=[3,2,4,3]
F=2 # 잃어버린 주사위 굴려서 나온 수
M=4 # 모든수의 합/개수
```

```
from sympy import Symbol, solve

def solution(A, F, M):    
    answer=0
    sum=0
    full_length = len(A)+F

    for i in A:
        sum += i

    x = Symbol('x')
    answer = abs(int(solve((sum+x)/(full_length), x)[0]))/2

    return [int(answer),int(answer)]

```


3. CV 알고리즘 만들기

```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(indices, K):
    # write your code in Python 3.6
    pass
```

```
for i in range(len(indices)):
    print(indices[i:i+K])
```
