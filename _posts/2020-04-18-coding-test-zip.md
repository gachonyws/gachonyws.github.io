---
title: "Coding Test: 행렬의 덧셈"
date: 2020-04-18
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

### 행렬의 덧셈

---

#### 문제 설명

행렬의 덧셈은 행과 열의 크기가 같은 두 행렬의 같은 행, 같은 열의 값을 서로 더한 결과가 됩니다. 2개의 행렬 arr1과 arr2를 입력받아, 행렬 덧셈의 결과를 반환하는 함수, solution을 완성해주세요.

#### 제한조건

- 행렬 arr1, arr2의 행과 열의 길이는 500을 넘지 않습니다.

#### 입출력 예

| arr1 |	arr2 |	return |
| [[1,2],[2,3]] |	[[3,4],[5,6]] |	[[4,6],[7,9]] |
| [[1],[2]] |	[[3],[4]] |	[[4],[6]] |

---

```python
def solution(arr1, arr2):
    answer = [[]]

    for i,j in zip(arr1,arr2):
        temp_list = []
        for k in range(len(i)):
            temp_list.append(i[k] + j[k])

        answer.append(temp_list)

    return answer
```

최초 시도시 zip을 한번만 사용하여 한번만 쓰이고 버리는 temp_list를 사용하여 아마 자원낭비(?)가 있었을 것이다.

```python
def solution(arr1,arr2):
    answer = [[c+d for c,d in zip(a,b)] for a,b in zip(arr1,arr2)]
    return answer
```

zip을 2회 사용하여 풀이. 생각보다 더 직관적이라고 느껴짐. zip은 pandas등에서도 데이터를 여러가지 형태로 묶을 때 많이 쓰이게 되는 것이니 기억해둘 필요가 있다.
