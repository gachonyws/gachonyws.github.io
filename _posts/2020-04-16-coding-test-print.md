---
title: "Coding Test: 직사각형 별찍기"
date: 2020-04-16
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
  - zip
published: true
---

### 직사각형 별찍기

---

#### 문제 설명

이 문제에는 표준 입력으로 두 개의 정수 n과 m이 주어집니다.
별(\*) 문자를 이용해 가로의 길이가 n, 세로의 길이가 m인 직사각형 형태를 출력해보세요.

#### 제한조건

- n과 m은 각각 1000 이하인 자연수입니다.


#### 입출력 예

입력
```
5 3
```
출력
```
*****
*****
*****
```
---

```python
a, b = map(int, input().strip().split(' '))
row = ''
row = row.zfill(a).replace('0','*')

for i in range(b):
    print(row)
    return answer
```

zfill로 a만큼(가로의 길이) 0으로 채운 후 별모양으로 바꿔준다. 그 후에 b(세로의 길이) 만큼 print 하면 끝.

```python
a, b = map(int, input().strip().split(' '))
answer = ('*'*a +'\n')*b

answer = answer.rsplit('\n',1) # 공백제거를 위한 코드
answer.remove('') # 공백제거를 위한 코드

print(answer[0])
```

2번줄의 ```answer = ('*'*a +'\n')*b``` 에서 '\n'이 마지막 줄까지 적용되어 공백 한줄이 생기는 불상사가 발생하지만 채점에서는 이상없이 통과되었다. 하지만 공백이 생기는 것은 출력물의 총 길이가 바뀐다던가 하는 등의 문제가 될 소지가 있어 마지막 '\n' 을 공백으로 바꿔준 후 지워줌.
