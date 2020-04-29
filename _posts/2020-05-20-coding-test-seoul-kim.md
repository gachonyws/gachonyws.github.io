---
title: "Coding Test: 서울에서 김서방 찾기"
date: 2020-05-20
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

### 서울에서 김서방 찾기

---

#### 문제 설명

String형 배열 seoul의 element중 "Kim"의 위치 x를 찾아, "김서방은 x에 있다"는 String을 반환하는 함수, solution을 완성하세요. seoul에 "Kim"은 오직 한 번만 나타나며 잘못된 값이 입력되는 경우는 없습니다.

#### 제한사항

- seoul은 길이 1 이상, 1000 이하인 배열입니다.
- seoul의 원소는 길이 1 이상, 20 이하인 문자열입니다.
- "Kim"은 반드시 seoul 안에 포함되어 있습니다.

#### 입출력 예

| ["Jane", "Kim"] | "김서방은 1에 있다" |

---

```python
def solution(seoul):
    answer = ''
    answer = str(seoul.index('Kim'))

    return  '김서방은 ' + answer + '에 있다'


```

1. str의 index 활용 쉬운 풀이
