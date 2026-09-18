# AppWorld full Test-Normal / Test-Challenge evaluation

This page publishes aggregate evaluation results only. Raw task statements, world snapshots,
simulated credentials, execution archives, and Phoenix traces remain local. The
[official leaderboard submission](https://github.com/StonyBrookNLP/appworld-leaderboard/pull/23)
contains encrypted bundles, not the raw material.

## Protocol

- AppWorld 0.1.3.post1; full Test-Normal (168 tasks) and Test-Challenge (417 tasks).
- A single recorded run per task. Unscored tasks count as non-passes in the denominator.
- Official `world.evaluate()` results determine success. Agent self-reports and successful
  infrastructure spans do not count as task success.
- Cloud planning and execution used Qwen3.8-Flash and GLM-5.3-Flash, with no Max/Pro tier.
  Medium reasoning was limited to Scope Resolver; most other roles used low or disabled reasoning.

## Results

| Metric | Test-Normal | Test-Challenge |
| --- | ---: | ---: |
| Task-goal completion / Pass@1 | **83.3% (140/168)** | **75.5% (315/417)** |
| Scenario-goal completion, official workflow | 73.2% | 51.8% |
| Failed / unscored | 27 / 1 | 99 / 3 |
| Prompt-cache hit ratio, task median / task mean | 76.2% / 75.6% | 73.8% / 72.5% |
| Estimated API cost per task, P50 / P90 / P95 | ¥0.13 / ¥0.29 / ¥0.36 | ¥0.19 / ¥0.41 / ¥0.47 |
| End-to-end latency, P50 / P90 / P95 | 281 / 553 / 662 s | 352 / 659 / 783 s |

Cost is reconstructed from observable provider usage and public prices; it is **not** a provider
invoice or an official leaderboard metric. The known-cost totals (¥27.84 Normal and ¥97.52
Challenge) are lower bounds because 33 and 37 requests, respectively, lacked usage metadata.
Cache percentages above are per-task statistics rather than a token-weighted global ratio.

The benchmark is evidence for these frozen runs, not a claim that later code revisions, provider
versions, or stochastic decodes will reproduce the same score.
