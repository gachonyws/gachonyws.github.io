---
title: "Coding Test: 수박수박수박수박수박수?"
date: 2020-05-22
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

### 수박수박수박수박수박수?

---

#### 문제 설명

길이가 n이고, 수박수박수박수....와 같은 패턴을 유지하는 문자열을 리턴하는 함수, solution을 완성하세요. 예를들어 n이 4이면 수박수박을 리턴하고 3이라면 수박수를 리턴하면 됩니다.

#### 제한사항

- n은 길이 10,000이하인 자연수입니다.


#### 입출력 예

| 3 | "수박수" |
| 4 | "수박수박" |

---

```python
def solution(n):
    answer = ''
    answer = answer.zfill(n)

    for i in range(n):
        answer = answer.replace('0','수',1)
        answer = answer.replace('0','박',1)

    return answer

```
