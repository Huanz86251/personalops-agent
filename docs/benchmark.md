# AppWorld Test-N Normal benchmark

This page publishes aggregate evaluation results only. Raw task statements, world snapshots,
simulated credentials, execution archives, and Phoenix traces remain local.

## Protocol

- Dataset: AppWorld Test-N, Normal split, 168 tasks; Challenge tasks excluded.
- One attempt per task (`Pass@1`); no failed-task reruns or result replacement.
- The fixed manifest was split into two seeded halves and later recombined without overlap.
- Official `world.evaluate()` results determine success. Agent self-reports and successful
  infrastructure spans do not count as task success.
- Cloud planning and execution used Qwen3.8-Flash and GLM-5.3-Flash. No Max/Pro tier and no
  high-reasoning mode was used.

## Results

| Metric | Result |
| --- | ---: |
| Pass@1 | **74.4% (125/168)** |
| Failed | 36 |
| Unscored, conservatively counted as failure | 7 |
| Task-average prompt-cache hit rate | **78.9%** |
| Total known public-price estimate | ¥29.02 lower bound |
| Cost per task, P25 / P50 / P90 / P95 | ¥0.074 / **¥0.112** / ¥0.355 / ¥0.426 |
| End-to-end latency, P25 / P50 / P90 / P95 | 143.0s / **215.1s** / 645.9s / 722.8s |
| Model calls per task, P25 / P50 / P90 / P95 | 17 / **21** / 51.3 / 60 |

The cost is reconstructed from observable provider usage and public prices. Sixteen requests
lacked usage or cost metadata, so ¥29.02 is a lower bound rather than a provider invoice.
The 78.9% cache figure is the arithmetic mean of each task's cache-hit ratio; the token-weighted
aggregate is 78.6%.

The benchmark is evidence for this pinned configuration, not a claim that every future provider
version or random decode will reproduce the same score.
