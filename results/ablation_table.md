| Config | Task success | Prompt tokens / task | Mean wrong-tool calls |
| --- | ---: | ---: | ---: |
| bm25_compact (best) | 100.0% | 1618.9 | 0.00 |
| hierarchical_compact | 100.0% | 1618.9 | 0.00 |
| embed_compact | 100.0% | 1631.4 | 0.00 |
| bm25 | 100.0% | 1666.0 | 0.00 |
| hierarchical | 100.0% | 1666.0 | 0.00 |
| embed | 100.0% | 1678.6 | 0.00 |
| all_tools_compact | 0.0% | 92509.0 | 1.80 |
| all_tools (baseline) | 0.0% | 92559.9 | 1.80 |

| Retriever | Recall@5 on test |
| --- | ---: |
| bm25 | 1.000 |
| embed | 1.000 |
| hierarchical | 1.000 |

| Criterion | Value | Threshold | Passed |
| --- | ---: | ---: | --- |
| Task success on held-out test templates (best configuration) minus all-tools baseline (points) | 100.0000 | >= 20.0 | True |
| Tool retrieval Recall@5 for the gold next tool | 1.0000 | >= 0.9 | True |
| Prompt token reduction vs all-tools baseline | 0.9825 | >= 0.6 | True |
| Side-effecting tool calls without required confirmation | 0.0000 | <= 0.0 | True |
| Schema-valid tool calls | 1.0000 | >= 0.98 | True |
| Test coverage | 0.9062 | >= 0.8 | True |
