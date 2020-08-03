---
title: "Coding Test: 자연수 뒤집어 배열로 만들기"
date: 2020-04-25
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

###  자연수 뒤집어 배열로 만들기

---

#### 문제 설명

자연수 n을 뒤집어 각 자리 숫자를 원소로 가지는 배열 형태로 리턴해주세요. 예를들어 n이 12345이면 [5,4,3,2,1]을 리턴합니다.

#### 제한사항

- n은 10,000,000,000이하인 자연수입니다.


#### 입출력 예

| n |	return |
| 12345	 |5,4,3,2,1] |

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
