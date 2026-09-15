"""Manual local smoke test for the fine-tuned MemOperator Write Gate."""

from pathlib import Path

from memory_write_gate import MemOperatorWriteGate


def main() -> None:
    gate = MemOperatorWriteGate(
        model_name="chris0809/memoperator-0.6b-memory-write-gate",
        cache_dir=Path(".models/memory_write_gate"),
        device="cpu",
        max_length=256,
        threshold=0.690976,
    )
    batches = [
        [
            "我以后希望你都用简洁的中文回答。",
            "谢谢你，讲得很好。",
            "我的项目预计十月底发布。",
        ],
        [
            "I prefer concise answers.",
            "What is the weather today?",
            "My name is Alice and I live in Boston.",
        ],
    ]
    for batch in batches:
        decisions = gate.classify(batch)
        print([(item.index, item.label) for item in decisions])


if __name__ == "__main__":
    main()
