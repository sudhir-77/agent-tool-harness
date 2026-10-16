# Agent harness for large tool spaces

This project is an agent harness for multi-step tasks over a simulated ecosystem of 300+ app tools. It plans one call at a time, retrieves a small set of tool schemas, compacts older observations, and blocks side-effecting tools until the call is confirmed. The reported agent is a local generative instruct model that emits the JSON tool call. Success is checked against the final app state.

## Setup

Python 3.11 or 3.12 on Windows:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

PyTorch is pinned to the CPU wheel (`--extra-index-url https://download.pytorch.org/whl/cpu` is the first line of `requirements.txt`).

## How to run

```
python run.py
python run.py --quick
python run.py --device cpu
```

`python run.py` generates the dataset under `data/`, downloads the Qwen2.5-1.5B-Instruct Q4_K_M GGUF (about 1 GB, cached under `data/models/`), evaluates the held-out test templates with that model, and writes `results/`. `python run.py --quick` runs the lexical comparison on 8 test tasks and skips the generative model and the embedding model so it finishes in a few minutes. Optional flags: `--skip-oracle`, `--skip-coverage`, `--device cpu`. The GGUF runtime uses the GPU when the installed llama.cpp wheel can offload layers, and otherwise uses CPU.

## Architecture

```mermaid
flowchart LR
  T[Task] --> H[Harness loop: plan, select tool, call, observe]
  H --> RET[Tool retrieval: BM25 / embeddings / hierarchical app-to-tool]
  H --> CTX[Context manager: compaction + memory]
  H --> PG[Permission gate for side effects]
  H --> M[Local Qwen instruct model, JSON tool call]
  H --> APPS[Simulated app state]
  APPS --> E[State-based eval + failure taxonomy]
```

The loop builds a prompt, keeps only the tool cards that fit in the model context (8,192 tokens, counted with the model tokenizer), asks the model for one JSON tool call, validates it with JSON Schema, retries once on a schema error, and runs the permission gate before any state change. Retrieval configurations place the top 5 tools in the prompt. The all-tools baseline places schemas in catalog order until the context budget is full and does not compact history. A second baseline places 5 tools chosen with a fixed seed.

## Design decisions

- **The reported agent is generative.** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) runs locally from `qwen2.5-1.5b-instruct-q4_k_m.gguf` through llama.cpp, greedy decoding, seed 0. It must emit the tool call as JSON. The harness validates that object and may retry once. Lexical grounding and the scripted oracle are extra rows in `lexical_episodes.csv` and `oracle_episodes.csv`, labeled in the `agent` column. They are not the reported agent.
- **Context limit is 8,192 tokens** for the generative model and retrieval depth is fixed at 5. The retriever used for the generative rows is chosen by Recall@5 on paraphrased dev instructions, not on the test split. The best row in the table is the evaluated config with the highest test success, then the fewest prompt tokens, then the name.
- **Dev and test templates are disjoint.** Even template indexes are dev and odd indexes are test. `data/manifest.json` stores checksums. If the generator changes after the files exist, the run stops instead of silently rewriting the split.
- **Prompt tokens for the reported agent are the model tokenizer counts** of the chat prompt actually sent (`usage.prompt_tokens`). The quick path still uses the whitespace tokenizer in `harness/tokens.py` because it does not load the GGUF.
- **Embeddings use frozen `sentence-transformers/all-MiniLM-L6-v2` vectors in a FAISS inner-product index.** The encoder is not trained on these tasks. It is unloaded before the GGUF loads unless embedding retrieval won the dev paraphrase comparison.
- **Side effects require `confirm=True`.** The gate returns `confirmation_required` and leaves state unchanged. The policy then repeats the call with confirmation. Executed side effects without confirmation are counted in `unconfirmed_side_effects`.

## Dataset

The tool catalog and the 400 tasks are generated in process by `apps/catalog.py` and `evals/generate.py` (seed 0). There is no third-party task corpus and no user data. License: generated locally by this project. The run writes `data/tools.json`, `data/tasks.json`, and `data/manifest.json`, then checks row counts (at least 300 tools, 400 tasks, 40 dev templates, 40 test templates, disjoint template ids) and sha256 checksums. The MiniLM `config.json` is downloaded with `requests` and checked against a pinned sha256. MiniLM weights are fetched by sentence-transformers into `data/hf_home` on the full run. MiniLM is Apache 2.0. The generative weights are `Qwen/Qwen2.5-1.5B-Instruct-GGUF` file `qwen2.5-1.5b-instruct-q4_k_m.gguf` (Apache 2.0), checked against a pinned sha256. On Windows the run also downloads the CUDA 12.4 cudart and cuBLAS redistributable DLLs (sha256 pinned) into `data/cuda/` so the llama.cpp wheel can use the GPU.

Dev templates are a diagnostic (`results/dev_diagnostics.json`). Reported success, token, and Recall@5 numbers use test templates only.

## Results

The table is written by `python run.py` from `results/metrics.json`. `results/episodes.csv` is the reported generative agent (`agent` column). Recompute success as the mean of `success` grouped by `config`. Recompute prompt tokens as the mean of `prompt_tokens`. Recompute schema-valid rate as the sum of `schema_valid_calls` divided by the sum of `n_steps` in `episodes.csv`. Recompute Recall@5 as the mean of `hit_at_5` in `results/retrieval.csv` for the retriever used by the best config. On test rows, `instruction_mentions_tool` is 0 when the instruction does not contain the tool id or action phrase. `lexical_episodes.csv` and `oracle_episodes.csv` are the labeled comparison agents. `gate_log.csv` records unconfirmed side effects.

<!-- RESULTS_START -->
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
<!-- RESULTS_END -->

Plots written next to the metrics file: `success_by_config.png`, `tokens_by_config.png`, `recall_at_5.png`, `failure_types.png`.

## Limitations

Dev instructions still use the canonical action phrase. Test instructions use a separate caller sentence that does not contain that phrase or the tool id (`instruction_mentions_tool` is 0 on those rows). The same caller sentence is written on the tool card, so retrieval is matching documented wording, not unseen slang. Slot values stay in `key=value` form so the arguments are recoverable from the text.

The generative model sees only the tools that fit in 8,192 tokens. The all-tools baseline keeps a catalog-order prefix of that window. A 1.5B model can still miss arguments, confirmation, or the right name when many cards are visible. If retrieval does not beat that baseline by 20 points, the results table says so.

Lexical grounding remains a deterministic comparison: on a phrase-style clause it selects the visible tool whose action phrase matches, and on a paraphrase it falls back to BM25 over the visible cards. The oracle may emit the labeled call only when that tool is visible. Neither is the reported agent.

History is placed after the tool list, so a tail truncation drops old observations before it drops the highest-ranked tools. Compaction therefore saves tokens on longer episodes and often leaves the visible tool set unchanged. Failure labels come from the episode trace and the state check: wrong tool, bad arguments, premature stop, loop, schema invalid, and budget exceeded.

## Tests

```
python -m pytest --cov=harness --cov=apps --cov=evals --cov-report=term-missing
```
