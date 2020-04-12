---
title: "Coding Test: x만큼 간격이 있는 n개의 숫자"
date: 2020-04-17
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
  - zip
published: true
---

### x만큼 간격이 있는 n개의 숫자

---

#### 문제 설명

함수 solution은 정수 x와 자연수 n을 입력 받아, x부터 시작해 x씩 증가하는 숫자를 n개 지니는 리스트를 리턴해야 합니다. 다음 제한 조건을 보고, 조건을 만족하는 함수, solution을 완성해주세요.

#### 제한조건

- x는 -10000000 이상, 10000000 이하인 정수입니다.
- n은 1000 이하인 자연수입니다.

#### 입출력 예

| x	| n	| answer |
| 2 | 5 | [2,4,6,8,10] |
| 4 |	3 |	[4,8,12] |
| -4 | 2| [-4, -8] |

---

```python
def solution(x, n):
    answer = []
    if x != 0:
        for i in range(x,x*(n+1),x):
            answer.append(i)
    if x ==0:
        for i in range(n):
            answer.append(0)

    return answer
```

애초부터 반복문의 시작점을 x부터 시작해서 x의 간격으로 넘어가게끔 함.

```python
def number_generator(x, n):
    return [i * x + x for i in range(n)]
```

한줄 버전.
이건 시작점을 0부터 n까지 시작점을 자리만 잡아주고 실제로 들어가는 값은 (i*x)+x로 x의 배수가 들어가도록 함
