---
title: "Coding Test: 체육복 "
date: 2020-05-10
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
  - 탐욕법(greedy)
---

### 체육복

---

#### 문제 설명

점심시간에 도둑이 들어, 일부 학생이 체육복을 도난당했습니다. 다행히 여벌 체육복이 있는 학생이 이들에게 체육복을 빌려주려 합니다. 학생들의 번호는 체격 순으로 매겨져 있어, 바로 앞번호의 학생이나 바로 뒷번호의 학생에게만 체육복을 빌려줄 수 있습니다. 예를 들어, 4번 학생은 3번 학생이나 5번 학생에게만 체육복을 빌려줄 수 있습니다. 체육복이 없으면 수업을 들을 수 없기 때문에 체육복을 적절히 빌려 최대한 많은 학생이 체육수업을 들어야 합니다.

전체 학생의 수 n, 체육복을 도난당한 학생들의 번호가 담긴 배열 lost, 여벌의 체육복을 가져온 학생들의 번호가 담긴 배열 reserve가 매개변수로 주어질 때, 체육수업을 들을 수 있는 학생의 최댓값을 return 하도록 solution 함수를 작성해주세요.

#### 제한사항

- 전체 학생의 수는 2명 이상 30명 이하입니다.
- 체육복을 도난당한 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없습니다.
- 여벌의 체육복을 가져온 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없습니다.
- 여벌 체육복이 있는 학생만 다른 학생에게 체육복을 빌려줄 수 있습니다.
- 여벌 체육복을 가져온 학생이 체육복을 도난당했을 수 있습니다. 이때 이 학생은 체육복을 하나만 도난당했다고 가정하며, 남은 체육복이 하나이기에 다른 학생에게는 체육복을 빌려줄 수 없습니다

#### 입출력 예

| n |	lost | reserve | return |
| 5	| [2, 4] |	[1, 3, 5] |	5 |
| 5	| [2, 4] |	[3] | 4 |
| 3	| [3] |	[1] | 2 |

#### 입출력 예 설명

입출력 예

예제 #1
1번 학생이 2번 학생에게 체육복을 빌려주고, 3번 학생이나 5번 학생이 4번 학생에게 체육복을 빌려주면 학생 5명이 체육수업을 들을 수 있습니다.

예제 #2
3번 학생이 2번 학생이나 4번 학생에게 체육복을 빌려주면 학생 4명이 체육수업을 들을 수 있습니다.


[출처](http://hsin.hr/coci/archive/2009_2010/contest6_tasks.pdf)

---

```python
def solution(n, lost, reserve):
    answer = 0
    reserve_set = set(reserve) - set(lost)
    lost_set = set(lost) - set(reserve)

    for i in reserve_set:
        if i-1 in lost_set:
            lost_set.remove(i-1)

        elif i+1 in lost_set:
            lost_set.remove(i+1)

    answer = n - len(lost_set)

    return answer
```

---

1. Set 자료형은 중복을 허용하지 않으며
2. Set 객체끼리 연산이 가능 (리스트로 중복허용X의 연산을 하려면 collection.Counter() 사용.)
3. i-1 (왼쪽)부터 확인하여 체육복을 빌려주어야 체육복이 남는데 빌려주지 못하는 상황 방지.

처음에 if만을 사용하여 조건을 걸어서 i-1 일때와 i+1 일때 모두 적용되어서 에러발생 -> if, elif로 제대로 적용되도록 수정
