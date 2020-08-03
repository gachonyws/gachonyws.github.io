---
title: "Coding Test: 정수 제곱근 판별"
date: 2020-04-23
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

###  정수 제곱근 판별

---

#### 문제 설명

임의의 양의 정수 n에 대해, n이 어떤 양의 정수 x의 제곱인지 아닌지 판단하려 합니다.
n이 양의 정수 x의 제곱이라면 x+1의 제곱을 리턴하고, n이 양의 정수 x의 제곱이 아니라면 -1을 리턴하는 함수를 완성하세요.

#### 제한사항

- n은 1이상, 50000000000000 이하인 양의 정수입니다.

#### 입출력 예

| n |	return |
| 121	 | 144 |
| 3	 | -1 |

#### 입출력 예 설명

입출력 예#1
121은 양의 정수 11의 제곱이므로, (11+1)를 제곱한 144를 리턴합니다.

입출력 예#2
3은 양의 정수의 제곱이 아니므로, -1을 리턴합니다.

---

```python
import math

def solution(n):
    answer = 0
    n_sqrt = math.sqrt(n)

    if n_sqrt.is_integer():
        answer = math.pow(n_sqrt+1,2)
    else:
        answer = -1

    return answer
```

math.sqrt()의 결과가 정수로 깔끔하게 나오면 해당 정수의 제곱근인 것이다. 이후 pow()를 통해 2제곱 리턴하도록 했다.
