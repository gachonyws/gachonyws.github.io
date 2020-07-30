---
title: "Coding Test: 모의고사"
date: 2020-05-03
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
  - 완전탐색
---

### 모의고사

---

#### 문제 설명

(조건)

수포자는 수학을 포기한 사람의 준말입니다. 수포자 삼인방은 모의고사에 수학 문제를 전부 찍으려 합니다. 수포자는 1번 문제부터 마지막 문제까지 다음과 같이 찍습니다.

- 1번 수포자가 찍는 방식: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...
- 2번 수포자가 찍는 방식: 2, 1, 2, 3, 2, 4, 2, 5, 2, 1, 2, 3, 2, 4, 2, 5, ...
- 3번 수포자가 찍는 방식: 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, 3, 3, 1, 1, 2, 2, 4, 4, 5, 5, ...

1번 문제부터 마지막 문제까지의 정답이 순서대로 들은 배열 answers가 주어졌을 때, 가장 많은 문제를 맞힌 사람이 누구인지 배열에 담아 return 하도록 solution 함수를 작성해주세요.

#### 제한사항

- 시험은 최대 10,000 문제로 구성되어있습니다.
- 문제의 정답은 1, 2, 3, 4, 5중 하나입니다.
- 가장 높은 점수를 받은 사람이 여럿일 경우, return하는 값을 오름차순 정렬해주세요.

#### 입출력 예

| answers |	return |
| [1,2,3,4,5] | [1] |
| [1,3,2,4,2] | [1,2,3] |

#### 입출력 예 설명

입출력 예 #1

- 수포자 1은 모든 문제를 맞혔습니다.
- 수포자 2는 모든 문제를 틀렸습니다.
- 수포자 3은 모든 문제를 틀렸습니다.
따라서 가장 문제를 많이 맞힌 사람은 수포자 1입니다.

입출력 예 #2

- 모든 사람이 2문제씩을 맞췄습니다.



```python
def solution(answers):
    answer = []
    score = [0,0,0]
    max_score = 0

    student1 = [1,2,3,4,5]
    student2 = [2,1,2,3,2,4,2,5]
    student3 = [3,3,1,1,2,2,4,4,5,5]

    for i in range(len(answers)):
        if answers[i] == student1[i % len(student1)]:
            score[0] += 1
        if answers[i] == student2[i % len(student2)]:
            score[1] += 1
        if answers[i] == student3[i % len(student3)]:
            score[2] += 1

    max_score = max(score)

    for i in range(0,3):
        if score[i] == max_score:
            answer.append(i+1)

    return answer
```

---

i가 1씩 증가하는 조건에서 각기 다른 길이의 패턴이 존재할 때!!!
각 패턴의 length를 <input으로 들어온 녀석의 길이로 나눈 나머지>로 나누면 증가하는 i에 맞춰서 패턴을 적용할 수 있다...
