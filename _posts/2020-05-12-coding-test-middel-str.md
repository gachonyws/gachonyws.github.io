---
title: "Coding Test: 가운데 글자 가져오기 "
date: 2020-05-12
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
visible: 0
tags:
  - level 1
---

### 가운데 글자 가져오기

---

#### 문제 설명

단어 s의 가운데 글자를 반환하는 함수, solution을 만들어 보세요. 단어의 길이가 짝수라면 가운데 두글자를 반환하면 됩니다.


#### 제한사항

- s는 길이가 1 이상, 100이하인 스트링입니다.

#### 입출력 예


| s |	return |
| abcde |	"c" |
| qwer | "we" |

---

```python
def solution(s):
    answer = ''

    if len(s) % 2 !=0: #2로 나누어 떨어지지 않는다면? 홀수
        answer = s[len(s)//2]
    else: # 2로 나누어 떨어진다면? 짝수
        answer = s[(len(s)//2 -1) :(len(s)//2+1)]

    return answer
```
