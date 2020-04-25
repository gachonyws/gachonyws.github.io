---
title: "Coding Test: 자릿수 더하기"
date: 2020-04-26
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

### 자릿수 더하기

---

#### 문제 설명

자연수 N이 주어지면, N의 각 자릿수의 합을 구해서 return 하는 solution 함수를 만들어 주세요.
예를들어 N = 123이면 1 + 2 + 3 = 6을 return 하면 됩니다.

#### 제한사항

- N의 범위 : 100,000,000 이하의 자연수

#### 입출력 예

| n |	answer |
| 123	 | 6 |
| 987	 | 24 |

#### 입출력 예 설명

입출력 예 #1
문제의 예시와 같습니다.

입출력 예 #2
9 + 8 + 7 = 24이므로 24를 return 하면 됩니다.
---

```python
def solution(n):
    answer = 0
    n = str(n)

    for i in range(len(n)):
        answer += int(n[i])

    return answer
```

int형 자료는 iterable 하지 않기 때문에 str로 변환 후 반복문을 돌리고 다시 int로 형변환.
