import ast
import json
import re
from pathlib import Path
import pytest
from pydantic import ValidationError
from skill_runtime.preparation import SkillChoice, _messages, _snapshot, skill_prompt
from skill_runtime import prepare_skills_sync
from test_skill_preparation import ProbeModel


def test_stackable_selection_and_explicit_empty_object():
    assert SkillChoice.model_validate({'skill_ids':[], 'reason':'none'}).skill_ids == []
    for invalid in ([], {}, {'skill_ids':['a','b','c','d'], 'reason':'too many'}):
        with pytest.raises(ValidationError): SkillChoice.model_validate(invalid)
    text=json.loads(_messages('general','AppWorld query',[])[0]['content'])['instruction']
    assert '选择零到3个' in text and '禁止返回裸数组' in text
    assert 'exclusive=true' in text and 'conflicts_with' in text


def test_invalid_selection_cannot_be_silently_injected_as_empty():
    snapshot=_snapshot('general','dynamic',[],[],'selection_error','ValidationError',1)
    with pytest.raises(ValueError,match='execution blocked'):skill_prompt(snapshot)
    assert skill_prompt(_snapshot('general','dynamic',[],[],'model','none',1))==''


def test_real_appworld_asset_reaches_execution_prompt():
    result=prepare_skills_sync(ProbeModel(selected=['appworld-execute-api']), role='general',
        task='AppWorld: query Spotify playlists', topics=['appworld'],
        tools=['appworld_discover','appworld_execute'],mode='dynamic')
    assert [s.name for s in result.selected]==['appworld-execute-api']
    assert '没有确认记录的需要新增' in result.selected[0].content
    assert '已新增、已修改或原本满足' in result.selected[0].content
    text=skill_prompt(result)
    assert 'apis.simple_note.show_note' in text
    assert '没有新依据则提交阻塞' in text
    assert '我创建的文件夹中、我标星的文档' in text
    assert 'A∩B' in text
    assert '只有“文件夹内文档或标星文档”才求A∪B' in text
    # Keep one complete discover/read/update/verify demonstration and only
    # a short contrast for a different application's login identifier.
    assert 'show_api_descriptions(app_name="simple_note")' in text
    assert 'apis.simple_note.search_notes' in text
    assert 'apis.simple_note.update_note' in text
    assert 'after["title"] == "评审安排"' in text
    assert 'simple_note.login` 文档说明 `username` 指账户邮箱' in text
    assert 'show_api_doc(app_name="phone", api_name="login")' in text
    assert 'username=profile["phone_number"]' in text
    assert 'Todoist' not in text
    assert '精确文档决定参数名及含义' in text


def test_phone_cross_app_contrast_keeps_two_valid_code_blocks():
    text = Path('skills/appworld/appworld-execute-api/SKILL.md').read_text(encoding='utf-8')
    section = text.split('### 新应用的简短对照：同名参数不能照搬', 1)[1].split('## 3.', 1)[0]
    blocks = re.findall(r'```python\n(.*?)\n```', section, flags=re.S)
    assert len(blocks) == 2
    for block in blocks:
        ast.parse(block)


def test_appworld_csv_export_skill_is_narrow_and_reaches_prompt():
    result = prepare_skills_sync(
        ProbeModel(selected=['appworld-csv-export']),
        role='general',
        task='AppWorld: export records to ~/backups/export.csv',
        topics=['appworld', 'csv', 'file-export'],
        tools=['appworld_discover', 'appworld_execute'],
        mode='dynamic',
    )
    assert [s.name for s in result.selected] == ['appworld-csv-export']
    text = skill_prompt(result)
    assert '最小必要引号' in text
    assert 'apis.file_system.create_file' in text
    assert "lstrip('\"')" in text
    assert '删除账户或执行其他不可逆' in text
    assert '验收器自己的类型错误' in text
    assert 'expected_pairs' in text and 'actual_pairs' in text
    assert '不要只在一边做 `.split("|")`' in text


def test_appworld_song_is_thin_and_stacks_with_execution_and_csv():
    selected = ['appworld-execute-api', 'appworld-csv-export', 'appworld-song']
    result = prepare_skills_sync(
        ProbeModel(selected=selected),
        role='general',
        task='AppWorld: choose a Spotify playlist by duration, then export CSV',
        topics=['appworld', 'spotify', 'csv'],
        tools=['appworld_discover', 'appworld_execute'],
        mode='dynamic',
    )
    assert {skill.name for skill in result.selected} == set(selected)
    song = next(skill for skill in result.selected if skill.name == 'appworld-song')
    assert not song.exclusive and not song.conflicts_with
    assert len(song.content) < 2500
    assert '换成同一单位' in song.content
    assert '不能单独推成“愿意重复听”' in song.content
    assert '2100 < 4500' in song.content
    assert 'study session today' in song.content
    assert 'required_seconds = day_minutes * 60' in song.content
    blocks = re.findall(r'```python\n(.*?)\n```', song.content, flags=re.S)
    assert len(blocks) == 1
    ast.parse(blocks[0])
    prompt = skill_prompt(result)
    for name in selected:
        assert f'[{name}]' in prompt


@pytest.mark.parametrize('role,selected,topics,tools', [
    ('scheduler', ['plan-appworld', 'plan-task-dependencies'],
     ['planning', 'appworld'], []),
    ('general', ['appworld-execute-api', 'appworld-csv-export'],
     ['appworld', 'csv'], ['appworld_discover', 'appworld_execute']),
    ('code', ['code-validate-schema', 'code-transform-files'],
     ['implementation', 'schema', 'files'], ['read_file', 'execute']),
    ('web', ['web-technical-docs', 'web-research-papers'],
     ['research', 'technical', 'papers'], ['web_search']),
    ('reviewer', ['review-schema-contract', 'review-file-output'],
     ['verification', 'schema', 'files'], ['read_file', 'execute']),
    ('step_reporter', ['review-appworld-results', 'report-execution-failure'],
     ['appworld', 'verification', 'recovery'], []),
    ('final_reviewer', ['final-review-evidence', 'final-review-appworld'],
     ['verification', 'final-review', 'appworld'], []),
])
def test_every_role_can_stack_compatible_skills(role, selected, topics, tools):
    result = prepare_skills_sync(
        ProbeModel(selected=selected), role=role, task='Combined task',
        topics=topics, tools=tools, mode='dynamic',
    )
    assert {skill.name for skill in result.selected} == set(selected)
    prompt = skill_prompt(result)
    for name in selected:
        assert f'[{name}]' in prompt


def test_exclusive_general_skill_cannot_stack_with_appworld():
    tools = ['attachment_to_text', 'appworld_discover', 'appworld_execute']
    topics = ['documents', 'appworld']
    for order, kept in [
        (['general-read-documents', 'appworld-execute-api'], 'general-read-documents'),
        (['appworld-execute-api', 'general-read-documents'], 'appworld-execute-api'),
    ]:
        result = prepare_skills_sync(
            ProbeModel(selected=order), role='general', task='Read AppWorld document',
            topics=topics, tools=tools, mode='dynamic',
        )
        assert [skill.name for skill in result.selected] == [kept]
        assert result.selection_method == 'model_conflict_filtered'
        assert 'stacking rules dropped' in result.reason
        with pytest.raises(ValueError, match='conflict'):
            prepare_skills_sync(
                ProbeModel(), role='general', task='Read AppWorld document',
                topics=topics, tools=tools, mode='fixed', fixed_ids=order,
            )


def test_explicit_pairwise_conflict_is_symmetric():
    for order, kept in [
        (['plan-appworld', 'plan-web-research'], 'plan-appworld'),
        (['plan-web-research', 'plan-appworld'], 'plan-web-research'),
    ]:
        result = prepare_skills_sync(
            ProbeModel(selected=order), role='scheduler', task='Mixed plan',
            topics=['planning', 'appworld', 'research'], tools=[], mode='dynamic',
        )
        assert [skill.name for skill in result.selected] == [kept]


def test_general_worker_keeps_short_write_before_classification_example():
    text = Path('prompts/workers/general_worker.md').read_text(encoding='utf-8')
    assert '没有确认记录的需要新增' in text
    assert '不要一边写入一边靠报错判断记录是否存在' in text
    assert '回读完整目标集合，而不只是刚修改的记录' in text


def test_scheduler_plans_all_final_state_branches_and_full_scope_readback():
    text = Path('prompts/planning/scheduler.md').read_text(encoding='utf-8')
    assert '记录不存在”不等于“已经满足' in text
    assert '读取、新增、修改和写后回读能力' in text
    assert '回读必须覆盖完整目标集合' in text
    assert '目标是 A∩B' in text
    assert '才是 A∪B' in text
    assert '按哪个稳定ID做交集/并集/差集' in text
    assert '绝对不能把“我拥有的容器中的已标记对象”' in text
    assert '上述复杂范围必须填写Step.target_selection' in text
    assert 'applies_to_sets或used_for_sets' in text


def test_scope_resolver_splits_independently_read_contexts():
    text = Path('prompts/planning/scope_resolver_system.md').read_text(encoding='utf-8')
    assert '必须拆成多条 required_context' in text
    assert '只有确实由同一次读取共同返回' in text
    assert 'EXPLICIT=用户明说' in text
    assert 'INFERRED=合理推测但执行前必须验证' in text
    assert '待处理 ∩ (朋友 ∪ 室友)' in text
    assert '"operation":"UNION"' in text
    assert '"resolution_status":"EXPLICIT"' in text
    assert 'schema 是表达接口，不是压缩目标' in text
    assert 'applies_to_sets 和 used_for_sets' in text
    assert "Tag incoming Venmo payments from my friends this week." in text
    assert "Review this month's Venmo transfers with my Venmo friends." in text
    assert '"source_system":"phone contact book"' in text
    assert '"source_system":"venmo"' in text
    assert 'friend永远等于Phone' in text


def test_appworld_scheduler_skill_preserves_predicate_ownership():
    text = Path('skills/scheduler/plan-appworld/SKILL.md').read_text(encoding='utf-8')
    assert '我创建的文件夹中、我标星的文档' in text
    assert '真正目标是A∩B' in text
    assert '按文档ID求交集' in text
    assert '真实接口名仍由执行者查文档' in text
    assert 'target_entity=document' in text
    assert "Tag incoming Venmo payments from my friends this week." in text
    assert "Review this month's Venmo transfers with my Venmo friends." in text
    assert 'phone.search_contacts' in text
    assert 'venmo.search_friends' in text
    assert '不得把API存在本身当作语义已经确定' in text
