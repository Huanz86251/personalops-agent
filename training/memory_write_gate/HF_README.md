---
license: apache-2.0
language:
- zh
- en
task_categories:
- text-classification
pretty_name: Memory Write Gate 20K
size_categories:
- 10K<n<100K
---

# Memory Write Gate 20K

This bilingual synthetic dataset trains a conservative long-term-memory write gate. Each user message is labeled as:

- `SAVE` (`label_id=1`): contains a durable fact, preference, relationship, decision, recurring rule, or future context that can help later conversations.
- `SKIP` (`label_id=0`): is a one-off request, current lookup, quoted or hypothetical content, deletion request, secret value, or temporary condition that should not become long-term memory.

The dataset contains 20,000 unique examples, balanced 10,000/10,000 across the two labels. It includes Chinese, English, and mixed-language messages. The `train`, `validation`, and `test` splits use disjoint seed families to prevent paraphrases of one seed from crossing splits.

| Split | Rows |
|---|---:|
| train | 14,200 |
| validation | 3,200 |
| test | 2,600 |

The label was fixed by 200 human-authored seed families. Qwen3.7-Flash, with thinking disabled, generated variations without choosing or changing the label. The second 10,000 examples target longer and harder messages, including messages that mix an immediate request with a durable fact.

Fields include `id`, `text`, `label`, `label_id`, `split`, `seed_id`, `category`, `language`, `domain`, and `difficulty`. Long-form rows also include requested and actual character lengths.

This is synthetic training data. It does not establish production accuracy, and users should evaluate privacy, false-save, and false-skip behavior on independently labeled real traffic before deployment. Generated rows were scanned for exact duplicates and obvious email, phone, passport, and API-key patterns before publication.
