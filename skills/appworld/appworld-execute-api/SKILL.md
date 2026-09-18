---
name: appworld-execute-api
description: 在 AppWorld 模拟应用中发现真实接口并查询或修改数据时选择；先区分文档发现与业务执行，再核验结果。
metadata:
  roles: "general"
  topics: "appworld"
  required-tools: "appworld_discover appworld_execute"
---

# 模拟应用执行策略

## 1. 确认接口与目标

- `appworld_discover` 只查目录、API名称和签名；`appworld_execute` 才查询或修改业务数据。两者使用同一任务世界中的现成 `apis`，不导入或创建 AppWorld。
- 每次调用填写简短 `source_refs`：复制支持本次动作的用户、RAG、状态或真实工具返回编号。引用说明来源，不保证来源正确；仍须核对适用范围。只有 `MODEL` 而没有任何现场来源时，不执行环境API。
- 发出业务代码前，核对其中每个 API 的准确名称、签名和参数来源是否已在当前上下文中得到确认；任何一个缺少依据，这一轮只用 `appworld_discover` 补齐，不发送依赖该缺项的业务代码。不要用“看起来应该存在”的方法作为默认值、后备分支或嵌套参数，也不要靠异常捕获试出接口名。已有依据的接口不重复查文档。
- 已有完整签名可直接使用；缺参数、凭据或返回字段时，只补查解决该缺项的文档或数据，齐备后执行下一步。调用一律使用关键字参数，账号信息取自 supervisor 的已文档化接口。
- 当前目标接口的精确文档决定参数名及含义；用户输入和本次真实返回提供参数值，不能反过来用手头已有的值猜接口含义。例：接口要求 `record_id`，文档示例为 `123`，应先查询目标记录的实际 ID，不能直接使用示例值；缺少必填值则补查或报告阻塞。
- 新证据触发重新确认：每次真实读取后，对照下一次操作所需的应用、API、关联键、凭据和参数值。若结果暴露了新缺口，或需要引入另一应用/接口，先暂停下一次业务调用，只补查缺项的目录、精确文档或真实数据。已查实的业务事实可以沿用；旧应用的 token、同名参数含义（如 `username`）和接口签名不能继承到新接口。依据按“应用 + API”区分；已有依据的不重复查，来源缺失或目标不唯一时不写入。
- 查询确认目标 ID；同名对象结合归属或关联记录区分。任务要求“全部”时处理完整分页，并按 ID 去重。

执行范围复核：写入前把每个条件对应的真实查询结果保存为ID集合，再按用户连接词计算。例：“我创建的文件夹中、我标星的文档”应以文件夹内文档ID集合A和标星文档ID集合B求A∩B；只有“文件夹内文档或标星文档”才求A∪B。接口没有直接返回最终集合时，分别查询后本地按ID计算。计划若把对象条件移到容器上，或算出的集合与原句不一致，以用户原句为准，停止写入并先修正集合。

目标集合 Action Card：当前 Step 带有 target_selection 时，每次业务调用都明确填写 action_phase。登录、取凭据等准备动作选 PREREQUISITE；第一次读取目标选 TARGET_READ。每个 operand 在同一层依次填写 read_api_reason、read_api、source_ref、requested_scope、coverage_reason、completion_condition；不要创建 coverage_plan 对象。READ_ONLY 在成功读取后直接 FINALIZE，并引用该读取回执，不得制造 TARGET_WRITE。MUTATION 的 TARGET_READ 成功后，Harness 会动态显示每个集合definition与所引来源中的真实接口描述；下一轮 TARGET_WRITE 必须在 binding_checks 中逐项先写reason，再填assessment，最后填set_id。全部为COMPLETE_MATCH才执行写入；出现INCOMPLETE_RESULT、WRONG_READ或CONTRACT_CONFLICT都不会执行代码，旧回执失效，同一Worker应重新查证并提交新的TARGET_READ。写入与回读通过binding_ref引用同一次冻结回执，set_bindings填[]；成功回读后再单独FINALIZE并引用TARGET_VERIFY。例：任务要求处理“项目文件夹内且已批准的报表”，若动态描述显示A确实返回文件夹内报表、B确实返回已批准报表，分别说明原因后选COMPLETE_MATCH；若B的描述实际是“用户收藏的报表”，应选WRONG_READ并重新读取，绝不能继续写入。

批量写入示例：用户要求“让模拟应用中属于我的全部申请最终为已确认”。先查全目标，并从真实返回取得 ID、归属和状态。没有确认记录的需要新增，已有但状态不对的需要修改，最终状态已经正确的才可跳过；缺 ID、归属或状态的先补查。不要把新增和修改混在一个试错循环里，也不要靠写入失败判断对象是否存在。写完后回读完整目标集合，确认每个目标都属于“已新增、已修改或原本满足”之一；有目标未归入这三组就不能报告完成。

Spotify 的短例子：目标歌曲编号必须先由真实查询取得。对每个目标调用 show_song_reviews(song_id=..., user_email=...)；已有本人评论但评分不对时，用返回的 song_review_id 调 update_song_review(review_id=..., rating=..., access_token=...)，没有本人评论时调 review_song(song_id=..., rating=..., access_token=...)。假设九首目标中本轮写了四首，完成后仍要对九首逐一再次调用 show_song_reviews，确认每首都有本人评论且评分正确；不能只检查刚写的四首，也不能拿 POST/PATCH 的成功消息当作最终状态。

## 2. 连续示例：查到真实信息后，再执行下一步

示例任务：将唯一标题为“项目备忘”的笔记改名为“评审安排”，正文不变。这不是当前任务。每段代码分轮发送，先读真实返回，再继续；示例中描述的成功条件不代表它已经发生。已有有效结果就复用，不重复查询。

**第一轮：尚不知道认证入口，用 `appworld_discover` 查真实目录。**
```python
print(apis.api_docs.show_api_descriptions(app_name="supervisor"))
print(apis.api_docs.show_api_descriptions(app_name="simple_note"))
```

**第二轮：目录实际列出了下列名称，继续用 `appworld_discover` 查签名。** 本例的 `simple_note.login` 文档说明 `username` 指账户邮箱；这只对该接口成立，不能把任意用户名或文档示例填进去。
```python
for app, name in [("supervisor", "show_profile"),
                  ("supervisor", "show_account_passwords"),
                  ("simple_note", "login")]:
    print(apis.api_docs.show_api_doc(app_name=app, api_name=name))
```

**第三轮：知道这个接口所需的账号标识和密码，但还没有实际值，改用 `appworld_execute` 读取。** 资料接口提供真实账号信息，密码接口返回按 `account_name` 区分的列表；只取目标应用的一条，不打印凭证。若之前已经拿到其中一项，只补查另一项。
```python
profile = apis.supervisor.show_profile()
accounts = apis.supervisor.show_account_passwords()
assert isinstance(profile, dict), "资料查询失败，停止"
assert isinstance(accounts, list), "密码查询失败，停止"
matches = [a for a in accounts if a.get("account_name") == "simple_note"]
assert len(matches) == 1 and matches[0].get("password"), "目标账户凭据缺失或不唯一"
print({"profile_ready": True, "target_credentials_ready": True})
```

**第四轮：当前接口文档确认 `username` 需要邮箱，才从真实资料中取邮箱；密码也须来自目标应用的真实返回。** 换应用时重新按其登录文档确定账号标识，不能照搬这里的邮箱。仅在实际返回 `access_token` 后才进入业务操作；失败就报告响应中的问题，不换猜测值重试。
```python
assert profile.get("email"), "此接口所需的真实账号邮箱缺失，停止"
auth = apis.simple_note.login(username=profile["email"], password=matches[0]["password"])
assert isinstance(auth, dict) and auth.get("access_token"), "登录未成功，停止业务操作"
token = auth["access_token"]
print({"authenticated": True})
```

**继续执行时：上轮已得到token，本任务的Python变量会保留，下一次调用直接复用。** 不要每次查询都重新取密码、登录；以下直接承接第四轮。接口是否需要认证，以实际签名为准：有必填access_token才传，文档明确无需认证的接口不登录、不额外塞token。不同应用的token不能混用；新执行环境没有旧变量，或真实返回明确指出凭据过期，才按已确认的认证流程重新获取。普通报错不等于登录失效。
```python
# 同一世界、同一应用，沿用第四轮已返回的token；不再次调用login。
# 确认show_note签名后，下一轮取得target_id即可使用：
# apis.simple_note.show_note(note_id=target_id, access_token=token)
# 文档查询接口本身不要求access_token；这段应放进appworld_discover。
print(apis.api_docs.show_api_doc(app_name="simple_note", api_name="show_note"))
```

**第五轮：登录成功，但还没有目标编号。** 用 `appworld_discover` 确认搜索、详情、修改的签名与返回字段，再进行下一轮查询。
```python
for name in ["search_notes", "show_note", "update_note"]:
    print(apis.api_docs.show_api_doc(app_name="simple_note", api_name=name))
```

**第六轮：文档确认搜索返回note_id和title，但不含正文。** 分页查询全部候选，确认标题唯一，编号只能来自实际结果。无匹配或多个匹配就停止，不造编号。
```python
rows = []
page_index = 0
while True:
    page = apis.simple_note.search_notes(
        access_token=token, query="项目备忘", page_index=page_index, page_limit=20)
    assert isinstance(page, list), "搜索未返回列表，停止并检查响应"
    rows.extend(page)
    if len(page) < 20:
        break
    page_index += 1
notes = {r["note_id"]: r for r in rows if r["title"] == "项目备忘"}
assert len(notes) == 1, "目标未找到或不唯一，停止修改"
target_id = next(iter(notes))
print({"note_id": target_id})
```
分页仍受当前执行预算约束；未查完就报告剩余部分，不能把部分结果当成全部。

**第七轮：编号已经拿到，不再重复搜索；正文还没有，所以查询详情保存原文。**
```python
before = apis.simple_note.show_note(note_id=target_id, access_token=token)
assert "content" in before and before.get("title") == "项目备忘", "详情不符，停止"
print({"note_id": target_id, "original_content_saved": True})
```

**第八轮：编号、原文、修改签名都已齐备，才只修改标题。**
```python
print(apis.simple_note.update_note(
    note_id=target_id, access_token=token, title="评审安排"))
```

**第九轮：修改返回成功消息，还不能证明要求全部满足。** 重新读取同一笔记，确认标题和正文；这次是验收新状态，不是重复查旧信息。
```python
after = apis.simple_note.show_note(note_id=target_id, access_token=token)
assert after["title"] == "评审安排" and after["content"] == before["content"]
print({"note_id": target_id, "title": after["title"], "content_unchanged": True})
```
通过后按当前Step提交证据。任何接口名、字段或必填值缺少实际依据，就补查已确认的文档或数据，绝对不能猜。已有完整文档可跳过文档查询；以上分轮是依赖顺序示例，不要求每次机械执行九轮。

### 新应用的简短对照：同名参数不能照搬

上例的邮箱只适用于已查实的 `simple_note.login`。若真实任务转向 Phone，先查目录（若尚未查过），再用 `appworld_discover` 查 `phone.login` 的精确文档；不能拿 `phone.search_contacts` 或旧应用的登录文档代替：
```python
print(apis.api_docs.show_api_doc(app_name="phone", api_name="login"))
```
本例文档说明 `username` 需要 `phone_number`，因此从已文档化的 Supervisor 真实返回中取手机号，并只取 Phone 对应的密码；缺值就停止，不能沿用上例的邮箱或 token：
```python
phone_accounts = [a for a in accounts if a.get("account_name") == "phone"]
assert profile.get("phone_number") and len(phone_accounts) == 1
assert phone_accounts[0].get("password"), "当前接口所需账号或凭据缺失，停止"
phone_auth = apis.phone.login(username=profile["phone_number"], password=phone_accounts[0]["password"])
assert isinstance(phone_auth, dict) and phone_auth.get("access_token"), "登录失败，停止"
```
`source_refs` 引用当前登录文档与真实账号回执；401 后先核对文档和值，不用同一组猜测值反复试。

## 3. 按结果分支

- 接口响应与预期不符或环境报错 → 先查相关文档和已有执行记录，取得明确依据后再修正调用；没有新依据则提交阻塞，不猜测入口或反复试验。
- 写入超时或响应缺失 → 先查当前状态；已生效则继续，未生效才重试，无法确认则报告未知。
- 部分成功 → 只继续未完成项。例如第5条已写入但响应丢失，确认后从第6条继续。
- 查询结果或断言不满足要求 → 保留差异和错误，不能报告完成。

## 4. 提交证据

按 General 现有报告协议提交：已完成项、查询依据、剩余项及阻塞错误；不输出凭证或无关整库数据。当前 Step 完成即提交，不越过分工继续其他 Step。
区分“未尝试”和“调用失败”；只报告实际观察到的错误，不把执行轮次耗尽写成接口超时或账户额度不足。

执行者自查不代替 Scheduler Final Review；工具返回成功不等于官方评测通过。隐藏 grader、答案及评测内部文件不可访问。
