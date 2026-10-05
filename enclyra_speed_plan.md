# Enclyra – Local LLM Speed Plan (summary)

## Setup
- MacBook M3, 8 GB RAM, Ollama
- Models: granite4:micro-h, granite4:micro, granite3.1-moe:3b
- Prompt is dynamic: SMR router picks modules per command (~3,700 tokens)

## Measured (TechStartup example)
| Model | Prompt read | Output write | Total |
|---|---|---|---|
| granite4:micro-h | 3,777 tok @ 121 tok/s = 31 s | 285 tok @ 11.7 tok/s = 24 s | ~57 s |
| granite4:micro | 3,777 tok @ 132 tok/s = 29 s | 234 tok @ 10.4 tok/s = 22 s | ~51 s |
| granite3.1-moe | ~510 tok/s | ~33 tok/s | fastest |

**Problem:** about half the time goes to re-reading the long prompt on every command.

## Key facts
- More RAM or more CPU cores does NOT make it faster. Memory bandwidth is the limit, and Ollama already uses the GPU.
- The prompt cache is big because it stores ~80 KB of model state per token, not the text (~300 MB per 3,700-token prompt at fp16, ~150 MB at q8).
- Caching all module combinations is impossible (too many). Only combinations that actually happen can be cached.
- On this single test, granite4:micro was more accurate than micro-h (micro-h skipped WriteToFile). Still to confirm with the QA harness.

## Decision: prompt STITCHING (no fine-tuning)
Works with any model and lets new modules be added without retraining.

How it works:
1. A = fixed header lines. Compute once and save `A.bin`.
2. For each module X: compute `A + X` and save only X's part, `X.bin`.
   A is the same every time (the model reads left to right), so it is stored once.
3. Per request: load A, then the needed modules. Move each module to its position (RoPE shift), then process only the dynamic tail (context + task).
4. Optional: recompute a few tokens at each join for better accuracy.
5. If A or a module's text changes, recompute that file.

Requirements:
- Must use **llama.cpp** (`llama-server` / `llama-cpp-python`), not Ollama. Ollama's GGUF files in `~/.ollama/models/blobs/` can be reused.
- Works on: granite4:micro ✅, granite3.1-moe ✅
- Does NOT work on granite4:micro-h ❌ (its Mamba layers keep one running state, so modules can't be split out)

## Expected speed (estimates)
| | Now | Stitching | + compact JSON |
|---|---|---|---|
| granite4:micro | ~51 s | ~25 s | ~18 s |
| granite3.1-moe | ~14.5 s | ~7.5 s | ~5.5 s |

After stitching, writing the output becomes the bottleneck.

## Quick wins (do first)
- `ollama ps` should show `100% GPU`. If it shows a CPU/GPU split, free memory.
- Env vars: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_MAX_LOADED_MODELS=1`
- num_ctx >= 8192 (3,777 + output is right at the 4,096 default, which risks truncation)
- Ask for compact single-line JSON output, and use Ollama `format` (JSON schema)

## Next steps
1. Run the quick wins.
2. Compare granite4:micro vs granite3.1-moe in the QA harness.
3. Build the stitching prototype (`stitch_cache.py`) and the plan comparison (`compare_plans.py`): full prompt vs stitched plans on the 16,782 unique logged prompts; target 98%+ identical.
4. Measure the real speed on the Mac.
5. Replace the Ollama call with `plan(task)`.

## QA baseline (before changes)
- 27,280 tests, median 24 s / p90 82 s per completed request
- Real pass rate ~86%; PATH failures are 48% of all failures
- Bugs found in the report: 185 "created a file then couldn't find it"; 572 "Nothing to summarize/write/copy" (output not passed to the next step)

## Other option (kept for later)
- Fine-tune (LoRA) so the prompt shrinks to ~150 tokens. Granite is Apache 2.0, so it can be used in a paid app.
- Not chosen because every new action would need retraining.
