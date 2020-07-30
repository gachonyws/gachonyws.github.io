---
title: "Riot Watcher API로 롤 감시기 만들기"
date: 2020-04-12
header:
  teaser: /assets/images/riot-watcher/log.png
  og_image: /assets/images/riot-watcher/log.png
categories:
  - project
tags:
  - API
  - Riot Developer
---

1. LoL Wathcer
   * Info
   * LoL Watcher
   * 환경 및 실습
 
---

### Info

* 원본

[Source code](https://github.com/pseudonym117/Riot-Watcher)


* Dependencies:
  - Python 3
  - riotwatcher <(https://github.com/pseudonym117/Riot-Watcher)>


* Reference
  - Riot Developer Portal <(https://developer.riotgames.com/)>
  - Spectator API <(https://riot-watcher.readthedocs.io/en/latest/riotwatcher/SpectatorApiV4.html#riotwatcher._apis.league_of_legends.SpectatorApiV4)>

---
### LoL Watcher

Riot games에서 제공하는 API를 활용하는 방법으로 게임중인 누군가를 확인하자!
API의 특성상 Riot Developer Portal(https://developer.riotgames.com/)에서 로그인 후 (롤 계정으로 로그인 가능) API키를 발급받는다.(API키는 24시간 후 만료되어 재발급 받아야 한다.)

| RATE LIMITS # 요청 limits 확인 |
| 20 requests every 1 seconds(s) |
| 100 requests every 2 minutes(s) |
{: .notice--info }

---

### 환경 및 실습

* 환경
  - Local(Python)
  - Jupyter Notebook (for test)

* 실습

```yaml
from riotwatcher import LolWatcher, ApiError

lol_watcher = LolWatcher('<API_KEY를 입력하세요>')

my_region = 'kr' # 지역은 북미에서 한국으로 변경해 주었습니다.
target = input('검색할 닉네임을 입력하세요:\n') # 키보드 입력을 받아 닉네임을 적도록 변경

me = lol_watcher.summoner.by_name(my_region, target)
print(me)

# all objects are returned (by default) as a dict
# lets see if i got diamond yet (i probably didnt)
my_ranked_stats = lol_watcher.league.by_summoner(my_region, me['id'])
print(my_ranked_stats)

# First we get the latest version of the game from data dragon
versions = lol_watcher.data_dragon.versions_for_region(my_region)
champions_version = versions['n']['champion']

# Lets get some champions
current_champ_list = lol_watcher.data_dragon.champions(champions_version)
print(current_champ_list)

# For Riot's API, the 404 status code indicates that the requested data wasn't found and
# should be expected to occur in normal operation, as in the case of a an
# invalid summoner name, match ID, etc.
#
# The 429 status code indicates that the user has sent too many requests
# in a given amount of time ("rate limiting").

try:
    response = lol_watcher.summoner.by_name(my_region, 'this_is_probably_not_anyones_summoner_name')
except ApiError as err:
    if err.response.status_code == 429:
        print('We should retry in {} seconds.'.format(err.headers['Retry-After']))
        print('this retry-after is handled by default by the RiotWatcher library')
        print('future requests wait until the retry-after time passes')
    elif err.response.status_code == 404:
        print('Summoner with that ridiculous name not found.')
    else:
        raise
```
*원본의 형태*

우선 원본의 코드를 통해 faker 선수의 아이디인 'hide on bush'를 입력하여 정보를 출력해 보겠습니다.

![복잡한 원본 코드의 결과](/assets/images/riot-wathcer/search-faker.png)
*다양한 정보가 JSON형태로 출력됩니다.*

---

사용자와 챔피언에 대한 정보는 감시기에서 필요가 없기 때문에 본격적인 수정. gameStartTime과 datetime.now()의 시간 차이를 이용해 플레이 중인지 확인

```yaml
from riotwatcher import LolWatcher
from datetime import datetime, timedelta
import time

lol_watcher = LolWatcher('RGAPI-b34c9615-2352-439e-8324-4326e28fbee1')
my_region = 'kr'
target = input('검색할 닉네임을 입력하세요:\n')

me = lol_watcher.summoner.by_name(my_region, target)

spectator = None

while True:
    print('[*] Checking...', "'"+target+"'", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    try:
        spectator = lol_watcher.spectator.by_summoner(my_region, me['id'])

        start_time = datetime.fromtimestamp(spectator['gameStartTime'] / 1000)

        if datetime.now() - start_time < timedelta(minutes=60):
            print('[!]' + target + ' is playing LoL! ', start_time.strftime('%Y-%m-%d %H:%M:%S'))
    except:
        pass

    time.sleep(5)
```


---

* 결과

실시간 감시를 웹서버에 올려 타겟의 게임 플레이가 감지되면 SMS,APP push 등의 알림기능을 넣는식으로 발전 가능할 것으로 보임.

| ![log](/assets/images/riot-watcher/log.png) |
|:--:|
| 실시간 로그 |
