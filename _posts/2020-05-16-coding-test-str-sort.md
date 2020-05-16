---
title: "Coding Test: 문자열 내 마음대로 정렬하기"
date: 2020-05-16
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

### 문자열 내 마음대로 정렬하기

---

#### 문제 설명

문자열로 구성된 리스트 strings와, 정수 n이 주어졌을 때, 각 문자열의 인덱스 n번째 글자를 기준으로 오름차순 정렬하려 합니다. 예를 들어 strings가 ['sun', 'bed', 'car']이고 n이 1이면 각 단어의 인덱스 1의 문자 'u', 'e', 'a'로 strings를 정렬합니다.

#### 제한사항

- strings는 길이 1 이상, 50이하인 배열입니다.
- strings의 원소는 소문자 알파벳으로 이루어져 있습니다.
- strings의 원소는 길이 1 이상, 100이하인 문자열입니다.
- 모든 strings의 원소의 길이는 n보다 큽니다.
- 인덱스 1의 문자가 같은 문자열이 여럿 일 경우, 사전순으로 앞선 문자열이 앞쪽에 위치합니다.

#### 입출력 예

| ['sun', 'bed', 'car'] | 1 | ['car', 'bed', 'sun'] |
| ['abce', 'abcd', 'cdx'] | 2 | ['abcd', 'abce', 'cdx'] |


#### 입출력 예 설명

입출력 예 1
'sun', 'bed', 'car'의 1번째 인덱스 값은 각각 'u', 'e', 'a' 입니다. 이를 기준으로 strings를 정렬하면 ['car', 'bed', 'sun'] 입니다.

입출력 예 2
'abce'와 'abcd', 'cdx'의 2번째 인덱스 값은 'c', 'c', 'x'입니다. 따라서 정렬 후에는 'cdx'가 가장 뒤에 위치합니다. 'abce'와 'abcd'는 사전순으로 정렬하면 'abcd'가 우선하므로, 답은 ['abcd', 'abce', 'cdx'] 입니다.


---

```python
def solution(strings, n):
    answer = []
    answer = sorted(sorted(strings), key=lambda word:word[n])

    return answer

```

1. sorted() 함수와 람다를 활용한 쉬운 풀이.
2. 최초 시도에서는 sorted(strings,key:lambda ~) 이런식으로 하여 입출력 예1은 해결이 가능했지만 입출력 예제2 'abcd' vs 'abce' 같은 녀석들 같은 경우엔 결과물을 한번더 sorted() 했어야 했다.
3. 따라서 처음부터 sorted(sorted(strings) ~ ) 처엄 정렬이 된 녀석을
넣어 코드를 한 줄로 줄여봄.
