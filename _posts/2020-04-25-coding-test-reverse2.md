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
| 12345	 | [5,4,3,2,1] |

---

```python
def solution(n):
    answer = []
    answer = list(map(int,reversed(str(n))))

    return answer
```

iterable 하게 str로 변환 후 뒤집어줌. 이후에는 map()을 사용하여 각 인덱스의 값들을 int로 변환하여 리스트로 만들어 마무리
