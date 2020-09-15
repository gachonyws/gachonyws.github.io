---
title: "Coding Test: 주식가"
date: 2020-05-28
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 2
published: true
---

### 124 나라의 숫자

---

#### 문제 설명

124 나라가 있습니다. 124 나라에서는 10진법이 아닌 다음과 같은 자신들만의 규칙으로 수를 표현합니다.
1. 124 나라에는 자연수만 존재합니다.
2. 124 나라에는 모든 수를 표현할 때 1, 2, 4만 사용합니다.
예를 들어서 124 나라에서 사용하는 숫자는 다음과 같이 변환됩니다.

|10진법|124 나라|10진법|124 나라|
|1|1|6|14|
|2|2|7|21|
|3|4|8|22|
|4|11|9|24|
|5|12|10|41|

자연수 n이 매개변수로 주어질 때, n을 124 나라에서 사용하는 숫자로 바꾼 값을 return 하도록 solution 함수를 완성해 주세요.

#### 제한 조건

- n은 500,000,000이하의 자연수 입니다.


#### 입출력 예

|n|	result|
|1|	1|
|2|	2|
|3|	4|
|4|	11|

---

```python
def solution(n):
    answer = ""

    if n <= 3:
        answer = '124'[n-1]

    else:
        number, remainder = divmod(n-1, 3)
        answer = solution(number) +'124'[remainder]

    return answer
```

1,2,4 세 가지 숫자의 패턴만 사용하기 때문에 3진법을 사용. 하지만 '124나라'에는 0이 없어 [0,1,2] 3진법이 아닌 [1,2,3] 3진법을 사용
