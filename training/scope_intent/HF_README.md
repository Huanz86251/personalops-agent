---
license: apache-2.0
language:
- zh
- en
task_categories:
- text-classification
pretty_name: Scope Intent Routing 20K
size_categories:
- 10K<n<100K
---

# Scope Intent Routing 20K

This bilingual synthetic dataset trains a high-recall router that decides whether a user request needs structured scope resolution before execution.

- `DIRECT_RESPONSE` (`label_id=0`): the request can be answered from the text already supplied by the user and needs no external state.
- `REQUIRES_SCOPE_CONTRACT` (`label_id=1`): the request needs tools, files, accounts, web access, databases, code execution, or other external reads or writes, so the system should first resolve objects, conditions, and operations.

The dataset contains 20,000 unique examples, balanced between the two labels. Its train/validation/test splits contain 14,000/3,000/3,000 rows and use disjoint seed families.

Labels were fixed by 80 human-authored seed families. Qwen3.7-Flash, with thinking disabled, generated paraphrases inside each fixed family and did not choose the label. The dataset is intended for conservative routing, where missing a request that needs scope resolution is more costly than occasionally invoking the resolver unnecessarily.

This is synthetic data and does not represent production traffic. Reported model metrics should be described as held-out synthetic-family results, not real-world accuracy.
