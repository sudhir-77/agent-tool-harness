# Resume notes

## Bullets

- Built an agent harness over a simulated 336-tool app ecosystem with tool retrieval, context compaction, and permission gates, raising held-out multi-step task success from 0.0% to 82.5% (hierarchical_compact versus the all-tools baseline, same local generative model).
- Reached 1.0 Recall@5 for next-tool retrieval while cutting prompt tokens per task by 96.8% versus putting tool schemas in the prompt up to the model context budget.
- Wrote state-based evals and a failure taxonomy (wrong tool, bad arguments, premature stop, loops) and labeled every held-out episode from the final app state.

## Skills and tools

Python, pydantic, JSON Schema, BM25 (rank-bm25), sentence-transformers, FAISS, llama.cpp, Qwen instruct, pytest, matplotlib.

## Summary

This project is a local agent harness that asks a generative instruct model for one JSON tool call at a time over a simulated 12-app ecosystem. It compares retrieval and context compaction with an all-tools prompt and a random five-tool prompt, and checks success from app state.

## GitHub description

Agent harness that retrieves tools, compacts context, and evaluates a local generative model on multi-step tasks over a 300+ tool simulated app ecosystem.
