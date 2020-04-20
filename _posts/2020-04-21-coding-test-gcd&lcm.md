---
title: "Coding Test: 최대공약수와 최소공배수"
date: 2020-04-21
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

###  최대공약수와 최소공배수

---

#### 문제 설명

두 수를 입력받아 두 수의 최대공약수와 최소공배수를 반환하는 함수, solution을 완성해 보세요. 배열의 맨 앞에 최대공약수, 그다음 최소공배수를 넣어 반환하면 됩니다. 예를 들어 두 수 3, 12의 최대공약수는 3, 최소공배수는 12이므로 solution(3, 12)는 [3, 12]를 반환해야 합니다.

#### 제한사항

- 두 수는 1이상 1000000이하의 자연수입니다.

#### 입출력 예

| n | m |	return |
| 3 | 12 | [3,12] |
| 2 | 5 | [1,10] |

---

1.  math() 사용

```python
import math

def solution(n, m):
    answer = []

    gcd = math.gcd(n,m) # greatest common divisor: 최대공약수
    lcm = (n*m)/gcd # least common multiple: 최소공배수

    answer.append(gcd)
    answer.append(lcm)

    return answer
```
greatest common divisor(gcd): 최대공약수, least common multiple(lcm):최소공배수를 구하는 문제. gcd는 math 클래스의 gcd() 함수가 있어 사용하였고 lcm은 두 수의 곱을 작은수로 나누어주면 최소 공배수를 구할 수 있다.


2. 유클리드 호제법 사용

```python
def gcdlcm(a, b):
    c, d = max(a, b), min(a, b)
    t = 1
    while t > 0:
        t = c % d
        c, d = d, t

        print(t,c,d)

    answer = [c, int(a*b/c)]

    return answer
```

유클리드 알고리즘은 2개의 자연수 또는 정식(整式)의 최대공약수를 구하는 알고리즘의 하나이다. 호제법이란 말은 두 수가 서로(互) 상대방 수를 나누어(除)서 결국 원하는 수를 얻는 알고리즘을 나타낸다.

2개의 자연수(또는 정식) a, b에 대해서 a를 b로 나눈 나머지를 r이라 하면(단, a>b), a와 b의 최대공약수는 b와 r의 최대공약수와 같다. 이 성질에 따라, b를 r로 나눈 나머지 r'를 구하고, 다시 r을 r'로 나눈 나머지를 구하는 과정을 반복하여 나머지가 0이 되었을 때 나누는 수가 a와 b의 최대공약수이다. 이는 명시적으로 기술된 가장 오래된 알고리즘으로서도 알려져 있다.

[유클리드_호제법](https://ko.wikipedia.org/wiki/%EC%9C%A0%ED%81%B4%EB%A6%AC%EB%93%9C_%ED%98%B8%EC%A0%9C%EB%B2%95)
