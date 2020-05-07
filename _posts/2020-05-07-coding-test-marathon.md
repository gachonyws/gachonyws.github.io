---
title: "Coding Test: 완주하지 못한 선수"
date: 2020-05-07
header:
  # teaser: /assets/images/coding-test/gamestar.png
  # og_image: /assets/images/page-header-teaser.png
categories:
  - coding test
  - python
tags:
  - level 1
  - hash
  - collections.Counter
---

### 완주하지 못한 선수

---

#### 문제 설명

수많은 마라톤 선수들이 마라톤에 참여하였습니다. 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.

마라톤에 참여한 선수들의 이름이 담긴 배열 participant와 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때, 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.

#### 제한사항

- 마라톤 경기에 참여한 선수의 수는 1명 이상 100,000명 이하입니다.
- completion의 길이는 participant의 길이보다 1 작습니다.
- 참가자의 이름은 1개 이상 20개 이하의 알파벳 소문자로 이루어져 있습니다.
- 참가자 중에는 동명이인이 있을 수 있습니다.

<table class="table">
        <thead><tr>
<th>participant</th>
<th>completion</th>
<th>return</th>
</tr>
</thead>
        <tbody><tr>
<td>[<q>leo</q>, <q>kiki</q>, <q>eden</q>]</td>
<td>[<q>eden</q>, <q>kiki</q>]</td>
<td><q>leo</q></td>
</tr>
<tr>
<td>[<q>marina</q>, <q>josipa</q>, <q>nikola</q>, <q>vinko</q>, <q>filipa</q>]</td>
<td>[<q>josipa</q>, <q>filipa</q>, <q>marina</q>, <q>nikola</q>]</td>
<td><q>vinko</q></td>
</tr>
<tr>
<td>[<q>mislav</q>, <q>stanko</q>, <q>mislav</q>, <q>ana</q>]</td>
<td>[<q>stanko</q>, <q>ana</q>, <q>mislav</q>]</td>
<td><q>mislav</q></td>
</tr>
</tbody>
      </table>


#### 입출력 예 설명

예제 #1
leo는 참여자 명단에는 있지만, 완주자 명단에는 없기 때문에 완주하지 못했습니다.

예제 #2
vinko는 참여자 명단에는 있지만, 완주자 명단에는 없기 때문에 완주하지 못했습니다.

예제 #3
mislav는 참여자 명단에는 두 명이 있지만, 완주자 명단에는 한 명밖에 없기 때문에 한명은 완주하지 못했습니다.

[출처](https://hsin.hr/coci/archive/2014_2015/contest2_tasks.pdf)

---

```python
def solution(participant, completion):
    answer = ''

    for i in range(len(participant)):    
        if participant[i] in completion:
            completion.remove(participant[i])
            participant[i] = ''

        else:
            continue

    for i in list(filter(lambda a: a!='',participant)):
        answer = i        

    return answer
```

---
1. 반복문을 돌면서 값 비교 후 겹치는 값을 찾아냄.
2. 겹치는 값은 ''로 대입하여 결국 리스트에는 여러개의 ''와 하나의 '사람이름'이 남음.
3. lambda로 ''를 없애고 출력 or 리스트.sort()[-1] 값을 출력시키면 정답

하지만 정확도에는 모두 통과성공했으나 효율성에서 실패가 떠버려 속도를 올리기 위해 collection의 Counter를 import하여 사용.. 원래 문제의 태그에 hash가 있던걸로 보아 본래 출제의도는 hash를 사용하라는 것 같지만 사용하지 않음.


```python
from collections import Counter

def solution(participant, completion):
    answer = list((Counter(participant)- Counter(completion)).keys())[0]

    return answer
```
