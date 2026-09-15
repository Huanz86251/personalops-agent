import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from skill_runtime.preparation import SkillChoice, _messages, _snapshot, skill_prompt
from skill_runtime import prepare_skills_sync
from test_skill_preparation import ProbeModel


def test_single_selection_and_explicit_empty_object():
    assert SkillChoice.model_validate({'skill_ids':[], 'reason':'none'}).skill_ids == []
    for invalid in ([], {}, {'skill_ids':['a','b','c','d'], 'reason':'too many'}):
        with pytest.raises(ValidationError): SkillChoice.model_validate(invalid)
    text=json.loads(_messages('general','AppWorld query',[])[0]['content'])['instruction']
    assert '单选' in text and '禁止返回裸数组' in text


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
    assert 'resolved_time_ranges' in text
    assert '2023-03-01至2023-03-31' in text
    assert 'Phone联系人簿中的关系标签' in text
    assert 'CSV兼容性' in text


def test_general_worker_keeps_short_write_before_classification_example():
    text = Path('prompts/workers/general_worker.md').read_text(encoding='utf-8')
    assert '没有确认记录的需要新增' in text
    assert '不要一边写入一边靠报错判断记录是否存在' in text
    assert '回读完整目标集合，而不只是刚修改的记录' in text
    assert 'resolved_time_ranges' in text
    assert '不能因为用户说“今年3月”就只截取月份' in text
    assert 'source_system和relationship' in text
    assert '不要用与写入器完全相反的同一套自定义解析逻辑自证' in text


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
    assert 'resolved_time_ranges都从ScopeContract原样保留' in text
    assert '包括其他年份同月' in text


def test_appworld_scheduler_skill_preserves_predicate_ownership():
    text = Path('skills/scheduler/plan-appworld/SKILL.md').read_text(encoding='utf-8')
    assert '我创建的文件夹中、我标星的文档' in text
    assert '真正目标是A∩B' in text
    assert '按文档ID求交集' in text
    assert '真实接口名仍由执行者查文档' in text
    assert 'target_entity=document' in text


def test_appworld_reviewers_reject_wrong_relationship_source_and_csv_self_check():
    step_text = Path('skills/appworld/review-appworld-results/SKILL.md').read_text(
        encoding='utf-8'
    )
    final_text = Path('skills/appworld/final-review-appworld/SKILL.md').read_text(
        encoding='utf-8'
    )
    for text in (step_text, final_text):
        assert 'required_context' in text
        assert 'Phone联系人' in text
        assert '自定义解析器' in text
        assert '同源自证' in text
