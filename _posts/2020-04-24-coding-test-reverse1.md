---
title: "Coding Test: 정수 내림차순으로 배치하기"
date: 2020-04-24
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

###  정수 내림차순으로 배치하기

---

#### 문제 설명

함수 solution은 정수 n을 매개변수로 입력받습니다. n의 각 자릿수를 큰것부터 작은 순으로 정렬한 새로운 정수를 리턴해주세요. 예를들어 n이 118372면 873211을 리턴하면 됩니다.

#### 제한사항

- n은 1이상 8000000000 이하인 자연수입니다.

#### 입출력 예

| n |	return |
| 118372	 | 873211 |

#### 입출력 예 설명

입출력 예 #1
문제의 예시와 같습니다.

입출력 예 #2
9 + 8 + 7 = 24이므로 24를 return 하면 됩니다.
---

```python
def solution(n):
    ls = list(str(n))
    ls.sort(reverse = True)
    return int("".join(ls))
```

iterable 하게 str로 변환 후 리스트로 만들어 준다. 이후 sort()로 뒤집어 준 후 각 값들을 join하여 하나의 값으로 만들어 준다.
