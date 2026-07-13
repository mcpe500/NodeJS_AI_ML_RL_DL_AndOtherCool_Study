# Handoff — needle.c (2026-07-14)

Branch: `needle.c/jul14`  
Remote: `origin` → https://github.com/mcpe500/NodeJS_AI_ML_RL_DL_AndOtherCool_Study  
Commits: `5d79fc5` framework+train+playground · `d9a895a` drop dbg junk  
PR: https://github.com/mcpe500/NodeJS_AI_ML_RL_DL_AndOtherCool_Study/pull/new/needle.c/jul14

## What shipped

- C11 framework `nd_*` (tensor, autograd, nn, optim, stream dataset, ckpt)
- Tests green: `make -C needle.c/tests`
- Models A/B/C full-epoch train on stream fixtures, f32, peak RSS ≤900
- Eval harness + greedy id infer
- **Playground web UI** stub: `make -C needle.c playground` → **0.0.0.0:7860**
- Docs: MASTER_PLAN, PLAN, program.txt, README (user vs Cactus gap honest)

## User try (web)

```bash
cd needle.c
make playground          # binds 0.0.0.0:7860
# browser: http://127.0.0.1:7860/  or http://<device-ip>:7860/
# PORT=8765 make playground   # if busy
# Ctrl+C stop
```

**Honest status:** form works; Generate shows **reference** tool JSON, not live model.  
Missing for Cactus parity: BPE, FC train data, `generate(query, tools)`.

Cactus real playground (other repo): https://cactuscompute.com/blog/needle

## Dev checks that work

```bash
make -C framework && make -C tests
make smoke MODEL=01-sanity
make train MODEL=02-pilot-6m
make train MODEL=03-full-26m
make -C eval && ./eval/evaluate A data/smoke.bin models/01-sanity/ckpts/sanity.nd
make infer
make chat-demo
```

## Observed train results (fixture vocab=64)

| Model | params | epochs | last loss | peak_rss | status |
|-------|--------|--------|-----------|----------|--------|
| A | 78k | 20 | ~0.55 | ~5 MB | keep |
| B | 4.0M | 10 | ~0.44 | ~65 MB | keep |
| C | 22M | 3 | ~1.0 | ~346 MB | keep |

Ckpts/logs/bins gitignored — regenerate via `make train`.

## Critical ownership rules

1. Free graph: `free(loss)` then `free(logits)` then batch leaves each step.
2. `nd_module_free` frees children first — `free_state` must **not** free children again (was abort root cause).
3. Peak RSS = VmHWM (never decreases); use `nd_current_rss_mb()` for live leak check.

## Next (not done)

1. BPE 8192 + real/synth function-calling JSONL → stream bins
2. Wire `generate(query, tools)` into playground (real OUTPUT)
3. Full attention backward (not just residual wq/wo path)
4. Act-ckpt recompute for C (accum already there)
5. KV-cache decode
6. Optional: merge PR to main

## Files map

```
needle.c/
  framework/     libneedle.a sources + headers
  tests/         unit + integration
  models/01-sanity|02-pilot-6m|03-full-26m/
  training/train_main.c   shared A/B/C entry
  eval/evaluate.c
  inference/{playground,chat_demo,generate}.c
  data/schema.md
  README.md  HANDOFF.md  program.txt  MASTER_PLAN.md
```
