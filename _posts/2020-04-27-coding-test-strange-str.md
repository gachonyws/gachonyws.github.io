---
title: "Coding Test: 이상한 문자 만들기"
date: 2020-04-27
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

### 이상한 문자 만들기

---

#### 문제 설명

문자열 s는 한 개 이상의 단어로 구성되어 있습니다. 각 단어는 하나 이상의 공백문자로 구분되어 있습니다. 각 단어의 짝수번째 알파벳은 대문자로, 홀수번째 알파벳은 소문자로 바꾼 문자열을 리턴하는 함수, solution을 완성하세요.

#### 제한사항

- 문자열 전체의 짝/홀수 인덱스가 아니라, 단어(공백을 기준)별로 짝/홀수 인덱스를 판단해야합니다.
- 첫 번째 글자는 0번째 인덱스로 보아 짝수번째 알파벳으로 처리해야 합니다.

#### 입출력 예

| s |	return |
| "try hello world"	 |	"TrY HeLlO WoRlD" |

#### 입출력 예 설명

"try hello world"는 세 단어 "try", "hello", "world"로 구성되어 있습니다. 각 단어의 짝수번째 문자를 대문자로, 홀수번째 문자를 소문자로 바꾸면 "TrY", "HeLlO", "WoRlD"입니다. 따라서 "TrY HeLlO WoRlD" 를 리턴합니다.

---

```python
def solution(s):
    answer = []
    s_sliced = s.split(' ')

    for word in s_sliced:
        temp=[]
        for i in range(len(word)):
            if i != ' ':
                if i % 2 ==0:
                    temp.append(word.replace(word[i],word[i].upper())[i])
                else:
                    temp.append(word.replace(word[i],word[i].lower())[i])
            else:
                temp.append(i)

        answer.append(''.join(temp))

    return ' '.join(answer)
```

처음 시도에서는 ```s.split()``` 으로만 문자열을 슬라이싱하여 진행하였고 8번 line의 ```if i != ' ':``` 부분이 없었다. 테스트 케이스에서는 잘 작동했지만 채점을 돌리면 30점 밖에 나오지 않음.
실제 채점과정에서 들어오는 케이스가 '   TrY    HeLlO   WoRlD ' 같은게 있었던 것 같아서 ```s_sliced = s.split(' ') ``` 와 같이 공백또한 모두 잘라서 결과물에 반영이 되도록 해주니 모두 통과되었다.
