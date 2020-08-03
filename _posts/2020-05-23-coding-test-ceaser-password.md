---
title: "Coding Test: 시저 암호"
date: 2020-05-23
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

### 시저 암호

---

#### 문제 설명

어떤 문장의 각 알파벳을 일정한 거리만큼 밀어서 다른 알파벳으로 바꾸는 암호화 방식을 시저 암호라고 합니다. 예를 들어 "AB"는 1만큼 밀면 "BC"가 되고, 3만큼 밀면 "DE"가 됩니다. "z"는 1만큼 밀면 "a"가 됩니다. 문자열 s와 거리 n을 입력받아 s를 n만큼 민 암호문을 만드는 함수, solution을 완성해 보세요.

#### 제한사항

- 공백은 아무리 밀어도 공백입니다.
- s는 알파벳 소문자, 대문자, 공백으로만 이루어져 있습니다.
- s의 길이는 8000이하입니다.
- n은 1 이상, 25이하인 자연수입니다.


#### 입출력 예

| s |	n |	result |
| "AB" | 1 | "BC"|
| "z" |	1 | "a" |
| "a B z" |	 4 | "e F d" |


---

```python
def solution(s, n):
    answer = ''
    s = list(s)

    for i in range(len(s)):
        if ord(s[i]) == 32:
            continue
          # 알파벳 a~z의 len: 26
        if s[i].islower():
            s[i] = chr((ord(s[i]) - 97 + n)%26 + 97) # 아스키코드 소문자 a: 97
        if s[i].isupper():
            s[i] = chr((ord(s[i]) - 65 + n)%26 + 65)# 아스키코드 대문자 A: 65

    answer = ''.join(s)

    return answer

```

아스키코드 상에서 이미 매핑되어있는 <정수:문자열>을 이용한 풀이.
아스키 코드 상에서 소문자 a는 97번이기 때문에 인풋으로 들어오는 문자열의 아스키코드를 ord()로 알아낸 후 소문자 a의 시작점인 97을 빼주고 인풋으로 들어온 n 만큼 더해준다. 그 이후에 대문자 알파벳의 총 길이인 26으로 나누어 주면 대문자 아스키 코드(97~122)의 인덱스 번호를 알아낼 수 있다.
