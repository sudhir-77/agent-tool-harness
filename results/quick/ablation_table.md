| Config | Task success | Prompt tokens / task | Mean wrong-tool calls |
| --- | ---: | ---: | ---: |
| bm25 (best) | 100.0% | 641.9 | 0.00 |
| bm25_compact | 100.0% | 641.9 | 0.00 |
| hierarchical | 100.0% | 641.9 | 0.00 |
| hierarchical_compact | 100.0% | 641.9 | 0.00 |
| all_tools (baseline) | 0.0% | 35736.1 | 1.00 |

| Retriever | Recall@5 on test |
| --- | ---: |
| bm25 | 1.000 |
| hierarchical | 1.000 |

| Criterion | Value | Threshold | Passed |
| --- | ---: | ---: | --- |
| Task success on held-out test templates (best configuration) minus all-tools baseline (points) | 100.0000 | >= 20.0 | True |
| Tool retrieval Recall@5 for the gold next tool | 1.0000 | >= 0.9 | True |
| Prompt token reduction vs all-tools baseline | 0.9820 | >= 0.6 | True |
| Side-effecting tool calls without required confirmation | 0.0000 | <= 0.0 | True |
| Schema-valid tool calls | 1.0000 | >= 0.98 | True |
