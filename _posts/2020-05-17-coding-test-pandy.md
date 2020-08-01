---
title: "Coding Test: 문자열 내 p와 y의 개수"
date: 2020-05-17
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

### 문자열 내 p와 y의 개수

---

#### 문제 설명

대문자와 소문자가 섞여있는 문자열 s가 주어집니다. s에 'p'의 개수와 'y'의 개수를 비교해 같으면 True, 다르면 False를 return 하는 solution를 완성하세요. 'p', 'y' 모두 하나도 없는 경우는 항상 True를 리턴합니다. 단, 개수를 비교할 때 대문자와 소문자는 구별하지 않습니다.

예를 들어 s가 "pPoooyY"면 true를 return하고 "Pyy"라면 false를 return합니다.

#### 제한사항

- 문자열 s의 길이 : 50 이하의 자연수
- 문자열 s는 알파벳으로만 이루어져 있습니다.

#### 입출력 예

| s | answer |
| 'pPoooyY' | True |
| 'Pyy' | False |


#### 입출력 예 설명

입출력 예 #1
'p'의 개수 2개, 'y'의 개수 2개로 같으므로 true를 return 합니다.

입출력 예 #2
'p'의 개수 1개, 'y'의 개수 2개로 다르므로 false를 return 합니다.

---

```python
from collections import Counter

def solution(s):
    answer = False    
    count = Counter(s)

    if count['p']+count['P'] == count['y']+count['Y']:
        answer = True
    if count['p']+count['P']== 0 and count['y']+count['Y'] == 0:
        answer = True

    return answer
```

1. collections의 Counter 를 사용하여 개수를 구하고
2. 조건문으로 해결
3. 다른 풀이

```python
# 문제가 소,대문자를 모두 세어야 하기 때문에 lower or upper로 대소문자 통일 후 str의 내장함수 count로 해결 가능
s = 'pPoooyY'
s.lower().count('p')
```
