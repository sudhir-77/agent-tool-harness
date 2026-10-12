# Resume notes

## Bullets

- Built an agent harness over a simulated 336-tool app ecosystem with tool retrieval, context compaction, and permission gates, raising held-out multi-step task success from 0.0% to 100.0% (bm25_compact versus the all-tools baseline, same lexical model).
- Reached 1.0 Recall@5 for next-tool retrieval while cutting prompt tokens per task by 98.3% versus putting every tool schema in the prompt.
- Wrote state-based evals and a failure taxonomy (wrong tool, bad arguments, premature stop, loops) and labeled every held-out episode from the final app state.

## Skills and tools

Python, pydantic, JSON Schema, BM25 (rank-bm25), sentence-transformers, FAISS, Hugging Face transformers, pytest, matplotlib.

## Summary

This project is a local agent harness that chooses tools for multi-step tasks over a simulated 12-app ecosystem. It compares BM25, embedding, and hierarchical retrieval against an all-tools prompt, with and without context compaction, and checks success from app state.

## GitHub description

Agent harness that retrieves tools, compacts context, and evaluates multi-step tasks over a 300+ tool simulated app ecosystem.
