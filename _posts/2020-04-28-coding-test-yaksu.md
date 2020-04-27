---
title: "Coding Test: 약수의 합"
date: 2020-04-28
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

### 약수의 합

---

#### 문제 설명

정수 n을 입력받아 n의 약수를 모두 더한 값을 리턴하는 함수, solution을 완성해주세요.

#### 제한사항

- n은 0 이상 3000이하인 정수입니다.

#### 입출력 예

| n |	return |
| 12 |	28 |
| 5 |	6 |

#### 입출력 예 설명

입출력 예 #1
12의 약수는 1, 2, 3, 4, 6, 12입니다. 이를 모두 더하면 28입니다.

입출력 예 #2
5의 약수는 1, 5입니다. 이를 모두 더하면 6입니다.

---

```python
def solution(n):
    answer = 0

    for i in range(1,n+1):
        if n%i == 0: answer += i

    return answer
```

1. str의 index 활용 쉬운 풀이
