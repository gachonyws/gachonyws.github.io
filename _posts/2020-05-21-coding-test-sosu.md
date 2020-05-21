---
title: "Coding Test: 소수 찾기"
date: 2020-05-21
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
published: true
---

### 소수 찾기

---

#### 문제 설명

1부터 입력받은 숫자 n 사이에 있는 소수의 개수를 반환하는 함수, solution을 만들어 보세요.

소수는 1과 자기 자신으로만 나누어지는 수를 의미합니다.
(1은 소수가 아닙니다.)

#### 제한사항

- n은 2이상 1000000이하의 자연수입니다.

#### 입출력 예

| 10 | 4 |
| 5 | 3 |

#### 입출력 예 설명

입출력 예 #1
1부터 10 사이의 소수는 [2,3,5,7] 4개가 존재하므로 4를 반환

입출력 예 #2
1부터 5 사이의 소수는 [2,3,5] 3개가 존재하므로 3를 반환

---

1. 단순무식하게 반복문 돌리기

```python
def solution(n):
    answer = 0

    for i in range(2,n+1):
        count = 0
        for j in range(1,i+1):    
            if i % j == 0:
                count += 1
        if count == 2:
            answer += 1

    return answer

```

1과 자기 자신만으로만 나누어 떨어지는 소수의 성질을 이용. 만약 5를 검증한다고 예를들면 5이하의 숫자 1~5를 모두 나누어 보면 나누어 떨어지는 것이 2개가 되면 소수인 것이다. 하지만 n이 10,000 까지는 금방 결과가 나왔지만  100,000부터는 연산효율이 매우 나빠 결과값이 나오질 않음.


 2.에라토스테네스의 체

 [에라토스테네스의 체](https://ko.wikipedia.org/wiki/%EC%97%90%EB%9D%BC%ED%86%A0%EC%8A%A4%ED%85%8C%EB%84%A4%EC%8A%A4%EC%9D%98_%EC%B2%B4)


 ```python
 def solution(n):
     answer = 0
     n_set = set(range(2,n+1))

     for i in range(2,n+1):
         if i in n_set:
             n_set -= set(range(i*2,n+1,i)) # 배수 삭제

         answer = len(n_set)

     return answer
 ```

 n_set = set(range(2,n+1))으로 나온 2~목표숫자까지의 수를 반복문을 통해 n_set -= set(range(i*2,n+1,i))로 배수들을 순차적으로 제거하면 소수만 남는다.
