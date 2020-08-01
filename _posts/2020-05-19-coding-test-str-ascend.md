---
title: "Coding Test: 문자열 다루기 기본"
date: 2020-05-19
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

### 문자열 다루기 기본

---

#### 문제 설명

문자열 s의 길이가 4 혹은 6이고, 숫자로만 구성돼있는지 확인해주는 함수, soltion을 완성하세요. 예를 들어 s가 a234이면 False를 리턴하고 1234라면 True를 리턴하면 됩니다.

#### 제한사항

- s는 길이 1 이상, 길이 8 이하인 문자열입니다.

#### 입출력 예

| "a234	" | false |
| "1234" | True |

---

```python
def solution(s):
    answer = False

    if len(s) == 4 or len(s) == 6:
        if s.isdigit():
            answer = True    

    return answer
```

1. isdigit(): 0~9 까지의 숫자로 이루어졌냐?
 - 3<sup>2</sup> 의 경우 True
2. isdecimal(): 숫자값 표현에 해당하는가? (아래 참고)
 - 3<sup>2</sup> 의 경우 False
3. isnumeric(): 숫자의 형태냐?
 - 3<sup>2</sup> 의 경우 True

1,2,3 모두 사용 가능하지만 차이점이 존재함. 따라서 만약 어떤 텍스트가 int 값으로 변환이 가능한지를 검사하고자 한다면 isdigit()을 사용해서는 안되며, isdecimal()을 써야 할 것이다.

3번의 numeric 하다는 것은 보다 넓은 의미인데, isdigit()은 단일 글자가 ‘숫자’ 모양으로 생겼으면 True를 반환한다고 했다. isnumeric()은 숫자값 표현에 해당하는 텍스트까지 인정해준다. 예를 들어 “½” 이런 특수문자도 isnumeric()에서는 True로 판정된다. (isdigit() 에서는 False)
