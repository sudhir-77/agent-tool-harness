| Config | Task success | Prompt tokens / task | Mean wrong-tool calls |
| --- | ---: | ---: | ---: |
| hierarchical_compact (best) | 82.5% | 827.2 | 0.15 |
| hierarchical | 81.0% | 870.9 | 0.15 |
| random5 | 0.5% | 662.5 | 1.74 |
| all_tools (baseline) | 0.0% | 26029.9 | 1.78 |

| Retriever | Recall@5 on test |
| --- | ---: |
| bm25 | 1.000 |
| embed | 1.000 |
| hierarchical | 1.000 |

| Criterion | Value | Threshold | Passed |
| --- | ---: | ---: | --- |
| Task success on held-out test templates (best configuration) minus all-tools baseline (points) | 82.5000 | >= 20.0 | True |
| Tool retrieval Recall@5 for the gold next tool | 1.0000 | >= 0.9 | True |
| Prompt token reduction vs all-tools baseline | 0.9682 | >= 0.6 | True |
| Side-effecting tool calls without required confirmation | 0.0000 | <= 0.0 | True |
| Schema-valid tool calls | 1.0000 | >= 0.98 | True |
| Test coverage | 0.8457 | >= 0.8 | True |
