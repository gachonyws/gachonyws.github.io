---
title: "Coding Test: 같은 숫자는 싫어 "
date: 2020-05-13
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

### 같은 숫자는 싫어

---

#### 문제 설명

배열 arr가 주어집니다. 배열 arr의 각 원소는 숫자 0부터 9까지로 이루어져 있습니다. 이때, 배열 arr에서 연속적으로 나타나는 숫자는 하나만 남기고 전부 제거하려고 합니다. 단, 제거된 후 남은 수들을 반환할 때는 배열 arr의 원소들의 순서를 유지해야 합니다. 예를 들면,

- arr = [1, 1, 3, 3, 0, 1, 1] 이면 [1, 3, 0, 1] 을 return 합니다.
- arr = [4, 4, 4, 3, 3] 이면 [4, 3] 을 return 합니다.
배열 arr에서 연속적으로 나타나는 숫자는 제거하고 남은 수들을 return 하는 solution 함수를 완성해 주세요.

#### 제한사항

- 배열 arr의 크기 : 1,000,000 이하의 자연수
- 배열 arr의 원소의 크기 : 0보다 크거나 같고 9보다 작거나 같은 정수

#### 입출력 예


| arr |	answer |
| [1,1,3,3,0,1,1] |	[1,3,0,1] |
| [4,4,4,3,3] |	[4,3] |
---

```python
def solution(arr):
    answer = []
    answer.append(arr[0])

    for i in range(len(arr)-1):
        if arr[i] != arr[i+1]:
            answer.append(arr[i+1])

    return answer
```

1. 가장 먼저 배열의 첫번째 값을 answer에 넣어놓은 후 반복문을 len(arr)-1 까지 수행. (len만큼 돌리면 배열의 가장 마지막 수가 들어가서 마지막 값의 인덱스+1과 비교하기 때문에 list index out of range 에러발생.)  --> range(1,len(arr)) 만큼 반복, 값의 비교를 arr[i] != arr[i-1] 이렇게 짜도 무방.
2. arr[i]의 값과 arr[i+1]의 값을 비교하여 answer에 추가.
