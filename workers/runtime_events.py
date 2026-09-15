"""Append changed harness state without rewriting earlier prompt prefixes."""

from hashlib import sha256

from langchain.messages import HumanMessage


def context_event(state, text: str, field: str):
    fingerprint = sha256(text.encode()).hexdigest()
    if state.get(field) == fingerprint:
        return None
    update = {field: fingerprint}
    if text or state.get(field):
        update["messages"] = [HumanMessage(
            content=("[状态更新]\n"
                     + (text or "此项已清空。")),
            name=field,
            additional_kwargs={"personalops_runtime_event": True},
        )]
    return update
