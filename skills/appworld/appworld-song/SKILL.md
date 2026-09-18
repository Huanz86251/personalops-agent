---
name: appworld-song
description: 在 AppWorld 音乐任务中，需要判断现有歌曲、专辑或播放列表能否覆盖活动时长，或需要决定循环、重复是否符合用户要求时选择；可与通用 AppWorld 执行及 CSV 导出 Skill 叠加。不用于仅查歌名、点赞或评论的任务。
metadata:
  roles: "general"
  topics: "appworld spotify music song playlist duration"
  required-tools: "appworld_discover appworld_execute"
---

# AppWorld Song：时长与循环

如果用户用“今天”等相对日期指向活动计划，先按任务世界的日期与星期定位本轮真实记录中的对应时长；用户已经直接给出时长，就不必额外查计划。只有需要用歌曲时长判断候选是否足够时，才核对请求时长与当前接口 `duration` 的单位；单位不同时再换成同一单位后比较，原本一致或无需比较时不走换算流程。换算仅用于判断，不改写 API 返回值或接口入参；单位不明时先查证，不猜。

若用户要一个现有播放列表或专辑覆盖时段，只累计本轮真实读取中符合用户限制、可用于这次播放的歌曲；不自行增加“其中每首歌都必须符合”的条件。默认每首歌只按一次播放计入，不为凑时长额外重复歌曲或整张列表。播放器支持循环只是功能，不代表用户允许重复；“不用中途换列表”也不能单独推成“愿意重复听”。只有用户明确允许循环或重复时，才可依据已查实的播放行为考虑重复覆盖。

英文合成例子（不是评测原题，数字不可照抄）："Play background music for my study session today. I want one playlist to last the whole session without switching. The schedule is in my notes." 先按任务世界的日期找笔记中今天那一行；若该行写 75 分钟、接口文档确认歌曲 `duration` 以秒返回，就在筛选任何候选前确定门槛为 `75 × 60 = 4500` 秒。`eligible_song_details` 必须来自本轮实际读取并符合用户限制，而不是文档样例。

```python
# 本例的 day_minutes 来自笔记中与任务世界当天日期/星期匹配的一行
# eligible_song_details 只包含本轮真实读取中符合用户限制的歌曲
required_seconds = day_minutes * 60
playlist_seconds = sum(song["duration"] for song in eligible_song_details)
enough_without_repeat = playlist_seconds >= required_seconds
print({"required_seconds": required_seconds,
       "playlist_seconds": playlist_seconds,
       "enough_without_repeat": enough_without_repeat})
```

若得到 2100 秒，`2100 < 4500`，这一候选不足 75 分钟；即使循环三遍在数字上够长，也不能在用户未允许重复时播放三遍并报告达标。继续只读查找其他现有候选；没有真实合格候选就说明未完成，不先播放一个不达标候选再解释它不够。除非用户明确授权，不为凑时长下载新歌、新建或改写播放列表，也不把用户范围外的歌曲塞进队列。播放后仍按已查实的接口回读实际队列与播放状态，确认队列中的歌曲仍符合用户限制且一轮时长达标；不能仅凭“开始播放”认定任务已满足。
