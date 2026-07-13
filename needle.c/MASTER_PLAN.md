# needle.c — MASTER PLAN (definitive, consolidated)

This document is the single source of truth for the needle.c project.
It merges architecture, reference guide, and experiment workflow into one
executable plan. Read this once; PLAN.md is the condensed loop reference,
PLAN.txt is the historical narrative, program.txt is the agent ops loop,
QUESTIONS.md is the decision journal.

---

## 0. PROJECT PITCH

Build **two products** in pure C11:

1. **Framework** (`nd_*`) — PyTorch-like tensors, autograd, modules, optimizers,
   stream dataloader, checkpoints, eager API + reusable captured graph.
2. **Model** — Needle-style **attention-only** encoder–decoder for single-shot
   tool-calling JSON (no FFN), with RoPE, ZCRMSNorm, gated residual, GQA,
   tied embeddings.

```text
Python/JAX reference ──parity──> C training framework
                                      │
                                      ▼
                              Attention-only model
                                      │
                                      ▼
                          CPU (Termux aarch64 first)
```

**Why:** study ML systems by building the stack; ship a usable function-calling
model that runs and trains on-device without Python at runtime.

**Where:** Termux on Android (aarch64, CPU only, no GPU).
**Measured (2026-07-14):** MemTotal ~7.1 GB, MemAvailable under load **~1.0–1.6 GB**,
Swap 4 GB, 8 cores (SM8350). Design for MemAvailable, never MemTotal.

**Method:** **Test-Driven Development (TDD)** is mandatory for framework and
model blocks. Write the failing test first, implement until green, then refactor.
No feature lands without a test that would have caught its absence.

---

## 0.1 HARDWARE AWARE (HARD — never violate)

Full float32 + full epoch counts are **required**. RAM safety comes from
**how** we load/train, not from gutting the model or precision.

| Cap | Value |
|-----|-------|
| Peak process RSS | ≤ **900 MB** |
| Dataset load | **stream only** (header + indices; fseek/fread per batch). Never malloc whole corpus. |
| Precision | **float32 full** — never drop to f16/bf16 for RAM on Termux path |
| Epochs | **full configured count** — never cut for RAM |
| Autograd | free graph every `opt->step` (`nd_tensor_free(loss)` releases subgraph) |
| Model A batch | default 4, max 8, min 1 |
| Model B batch | default 1, max 2, min 1 |
| Model C batch | **1 only** on Termux + activation checkpoint + grad accum 4–8 |
| OOM policy | log + halve batch once; still OOM → permanent `crash` |
| First backend | aarch64 FP32 reference C loops, single-thread |
| Parallelism | single-thread first; never N workers each holding a batch |

### Memory budget (float32, rough)

| Model | params | params MB | AdamW m+v | acts (rough) | peak |
|-------|--------|-----------|-----------|--------------|------|
| A Sanity d=64 L=2/2 T=64 B=4 | ~107k | 0.4 | 0.9 | ~3 | **~55 MB** |
| B Pilot d=256 L=8/6 T=256 B=1 | ~6.0M | 24 | 48 | ~59 | **~180 MB** |
| C Full d=512 L=12/8 T=512 B=1 | ~26.2M | 105 | 210 | ~335 | **~700 MB TIGHT** |

Model C on Termux: B=1 + activation checkpoint (recompute) + stream + free graph/step.
Desktop may raise batch.

### What stays full

- Full float32 math.
- Full epoch loops (no early exit for RAM).
- Full architecture within the model tier.
- Full eval on whole held-out set (streamed batches).

RAM levers only: stream data, free graph, microbatch/accum, checkpoint, lower B/T.

**Chunking / duration policy:** Prefer streamed micro-batches over full-corpus
RAM load. Longer wall-clock is acceptable when it prevents OOM; still optimize
(stream, free graph, act-ckpt on C) — do not make runs needlessly long. Never
trade away full float32 or full epoch counts for speed or RAM.


See also PLAN.md §0.1 and QUESTIONS.md Q0/Q4/Q5/Q7.

---

## 0.2 TEST-DRIVEN DEVELOPMENT (HARD — never skip)

TDD is the primary bug-prevention strategy. Order is non-negotiable:

```text
1. Write a failing test that specifies the contract.
2. Run it — confirm RED.
3. Implement the minimum code to pass.
4. Run it — confirm GREEN.
5. Refactor under green.
6. Commit.
```

### Rules

| Rule | Detail |
|------|--------|
| No feature without a test | Op, module, metric, or train step lands only with a test that fails without it |
| Red first | CI/local must show RED before the implementation commit when practical |
| Numerical parity | Attention, RoPE, ZCRMSNorm, GQA vs Python/JAX oracle (max abs err < tol) |
| Gradient checks | Analytical ≈ finite-difference for every new backward |
| Leak / RSS tests | N-step train loop; RSS must not climb unboundedly |
| Smoke before scale | Model A overfit GO/NO-GO before Pilot/Full data |
| Tests are frozen contracts | `tests/` and `eval/` are not editable in the experiment loop |
| Fast unit, slow integration | Unit tests < few seconds; oracle/parity may be slower, tagged |

### Test layers

```text
tests/
  unit/          tensor, autograd, shape, stream loader, optim
  numerical/     finite-diff gradients, oracle parity dumps
  smoke/         Model A overfit-32, RSS cap, graph free
  integration/   one full train step enc-dec, ckpt round-trip
eval/            held-out metrics harness (fixed) — not unit tests
tools/oracle/    Python golden generators (not runtime)
```

### Per-operator definition of done

Every op needs:

```text
forward CPU reference
backward CPU reference
shape inference
unit test (values + shapes)
numerical gradient test (where differentiable)
(optional later) SIMD/CUDA — only after CPU green
```

### GO/NO-GO gates (TDD-linked)

- M1–M3: unit + grad tests green before any model code.
- M4: oracle parity for attention block RED→GREEN before stacking layers.
- M7: smoke overfit must pass before any Pilot run.
- M9: eval harness has fixture tests (known JSON → known scores).
- Any experiment that breaks `make test` is an automatic `crash` / revert.

Headline: **bugs are cheaper in tests than in 6M-param training runs.**

---

## 1. GOALS & NON-GOALS

### In scope

* Pure C11 framework (`nd_*`): Tensor, autograd, nn modules, AdamW/SGD,
  stream dataloader, weight I/O, graph capture. No external libs initially.
* Float32, CPU, single-threaded first (NEON later stretch).
* Attention-only enc–dec tool-calling model at three sizes (A/B/C).
* BPE tokenizer 8192 (train in Python, load in C).
* Function-calling dataset (ID/EN), streamed binary format.
* Held-out eval: JSON parse rate, tool-name EM, arg key F1, arg value EM,
  **full-call EM**, no-tool accuracy, runtime RSS/toks.
* TDD suite: unit, numerical, smoke, integration.
* Comparison notes per model (wins/losses).

### Explicitly out of scope

* GPU / CUDA on this device (desktop stretch only).
* Dropping float32 or cutting epochs for RAM.
* Full-dataset RAM load.
* FFN/MLP inside attention layers (default stack is attention-only).
* Distributed / multi-process training.
* Production HTTP server.
* New C third-party deps for v1 (BLAS optional later).
* PyTorch/JAX as runtime (Python = oracle, synth, converter only).
* CMake as primary build (Makefile first).
* x86-64 as first platform (aarch64 Termux first).
* Editing frozen framework/kernels during experiment loop after green.
* INT4 QAT / BF16 / Muon before Model B baseline + capacity.

---

## 2. REPOSITORY STRUCTURE

```
needle.c/
├── README.md              Overview + (later) comparison table
├── MASTER_PLAN.md         This file
├── PLAN.md                Condensed loop reference
├── PLAN.txt               Verbose genesis narrative
├── program.txt             Autonomous experiment loop
├── QUESTIONS.md           Decision journal
├── Makefile               Top-level: framework, test, models, eval, smoke
│
├── framework/             READ-ONLY once tests green
│   ├── include/           needle.h, tensor.h, autograd.h, nn.h, optim.h,
│   │                      dataset.h, io.h, graph.h, util.h
│   ├── src/               matching .c
│   └── Makefile
│
├── kernels/               READ-ONLY once green (CPU ref first)
│   ├── cpu_ref/
│   ├── neon/              stretch
│   └── cuda/              desktop stretch only
│
├── tests/                 READ-ONLY in experiment loop (TDD home)
│   ├── unit/
│   ├── numerical/
│   ├── smoke/
│   ├── integration/
│   └── Makefile
│
├── models/                EDITABLE in experiment loop
│   ├── 01-sanity/         src/, Makefile, README.md, results.tsv
│   ├── 02-pilot-6m/
│   └── 03-full-26m/
│
├── tokenizer/             BPE load + Python train script
├── training/              trainer helpers (prefer thin; model owns loop knobs)
├── inference/             KV cache, generate, JSON decode
├── tools/                 oracle/, convert/, bench/
├── data/                  READ-ONLY once built (stream bins + synth scripts)
├── eval/                  READ-ONLY metrics harness
└── scripts/               smoke_test.sh, sanity, comparison
```

---

## 3. THE C FRAMEWORK (`nd_*`) — API SPEC

### 3.1 Design principles

* Pure C11. libc only for v1.
* Float32 only on Termux path. CPU only.
* All ops return `nd_tensor *`. Autograd when any input `requires_grad`.
* Refcount: last free drops data + grad_fn subgraph.
* No exceptions. `NULL` on alloc failure; log via `util.h`.
* Naming: `nd_tensor_*`, `nd_nn_*`, `nd_optim_*`, `nd_dataset_*`.
* Eager frontend; optional captured graph for train replay.
* **TDD:** every public function has at least one test in `tests/`.

### 3.2 Tensor struct

```c
#define ND_MAX_DIMS 8

typedef struct nd_node nd_node;

typedef struct nd_tensor {
    float    *data;
    float    *grad;
    int64_t   shape[ND_MAX_DIMS];
    int64_t   strides[ND_MAX_DIMS];
    int32_t   ndim;
    int32_t   size;          /* product of shape */
    bool      requires_grad;
    bool      owns_data;
    nd_node  *grad_fn;
    int       refcount;
} nd_tensor;
```

Invariants (enforced by tests):

- `size == product(shape[0..ndim))`.
- Contiguous row-major by default; views may non-own data.
- `grad_fn` set only on non-leaf outputs from ops with grad inputs.

### 3.3 Core API (contracts)

```c
/* lifecycle */
nd_tensor *nd_zeros(const int64_t *shape, int32_t ndim, bool requires_grad);
nd_tensor *nd_from_f32(const float *data, const int64_t *shape, int32_t ndim);
void       nd_tensor_retain(nd_tensor *t);
void       nd_tensor_free(nd_tensor *t);   /* frees subgraph if last ref */

/* shape */
nd_tensor *nd_view(nd_tensor *t, const int64_t *shape, int32_t ndim);
nd_tensor *nd_reshape(nd_tensor *t, const int64_t *shape, int32_t ndim);
nd_tensor *nd_transpose(nd_tensor *t, int32_t d0, int32_t d1);

/* ops (each: forward + backward + test) */
nd_tensor *nd_add(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_mul(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_matmul(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_softmax(nd_tensor *a, int32_t dim);
nd_tensor *nd_embedding(nd_tensor *weight, nd_tensor *ids_i32_as_storage);
nd_tensor *nd_cross_entropy(nd_tensor *logits, nd_tensor *target_ids);

/* autograd */
void nd_backward(nd_tensor *loss);
void nd_zero_grad_module(nd_module *m);

/* optim */
nd_optim *nd_adamw_create(nd_module *m, float lr, float wd);
void      nd_optim_step(nd_optim *o);
void      nd_optim_free(nd_optim *o);

/* stream dataset — HARDWARE AWARE */
nd_dataset *nd_dataset_open(const char *path);  /* header + index only */
int         nd_dataset_next_batch(nd_dataset *d, /* out buffers */, int batch);
void        nd_dataset_close(nd_dataset *d);

/* checkpoint */
int nd_checkpoint_save(const char *path, nd_module *m);
int nd_checkpoint_load(const char *path, nd_module *m);
```

### 3.4 Train-loop free contract (must be tested for RSS flatness)

```c
for (step = 0; step < steps; step++) {
    nd_dataset_next_batch(ds, &batch, B);
    nd_zero_grad_module(model);
    nd_tensor *logits = nd_module_forward(model, batch.src, batch.dst);
    nd_tensor *loss   = nd_cross_entropy(logits, batch.labels);
    nd_backward(loss);
    nd_optim_step(opt);
    nd_tensor_free(loss);    /* releases graph */
    nd_tensor_free(logits);
    /* batch buffers reused, not grown */
}
```

### 3.5 Captured graph (after eager green)

```c
nd_graph_begin_capture(ctx);
/* build once */
nd_graph *g = nd_graph_end_capture(ctx);
/* replay: bind inputs, forward, backward — no realloc of graph nodes */
```

### 3.6 nn modules (arch)

```c
nd_module *nd_linear(int in, int out, bool bias);
nd_module *nd_embedding_table(int vocab, int d);
nd_module *nd_zcrmsnorm(int d, float eps);
nd_module *nd_rope_gqa(int d, int n_q, int n_kv, bool causal);
nd_module *nd_gated_residual(/* attn child */);
nd_module *nd_encoder_stack(int d, int n_layers, int n_q, int n_kv);
nd_module *nd_decoder_stack(int d, int n_layers, int n_q, int n_kv);
nd_module *nd_needle_create(const nd_needle_config *cfg); /* tied emb */
```

---

## 4. ARCHITECTURE (Needle attention-only)

| Component | Full (C) | Pilot (B) | Sanity (A) |
|-----------|----------|-----------|------------|
| Arch | enc–dec, **no FFN** | same | same |
| d_model | 512 | 256 | 64 |
| vocab | 8192 BPE | 8192 | 8192 (or 512 toy) |
| encoder layers | 12 | 8 | 2 |
| decoder layers | 8 | 6 | 2 |
| query heads | 8 | 8 | 4 |
| kv heads | 4 | 4 | 2 |
| max length | 512 | 256 | 64 |
| pos | RoPE | RoPE | RoPE |
| norm | ZCRMSNorm | ZCRMSNorm | ZCRMSNorm |
| residual | gated | gated | gated |
| output | tied emb | tied emb | tied emb |

Formulas:

```text
ZCRMSNorm:  y = x * (1 + gamma) / sqrt(mean(x²) + eps)
Gated res:  y = x + sigmoid(gate) * attention(norm(x))
```

Encoder: bidirectional GQA. Decoder: causal self-attn + cross-attn to encoder.
Gate/gamma init near identity path (QUESTIONS.md Q9).

Rough full params ~26.2M core (HF may report ~30M with extras). Framework
must count params and report core vs retrieval-head vs other.

---

## 5. TOKENIZER

* BPE vocab **8192**.
* Train + export in Python (`tokenizer/train_bpe.py` later).
* C loads merges/vocab; encode/decode for inference demos.
* Training data is **pre-tokenized** to binary (C never requires Python).

TDD: encode→decode round-trip fixtures; unknown-byte handling.

---

## 6. DATASET (function-calling, streamed)

JSONL source form:

```json
{
  "query": "Set an alarm for 6:30 tomorrow morning",
  "tools": [{ "name": "create_alarm", "description": "...", "parameters": {...} }],
  "answer": [{ "name": "create_alarm", "arguments": { "time": "06:30" } }]
}
```

Composition requirements: positive single-tool, distractors, no-tool, missing
required, optional args, copy-heavy, paraphrase, tool-order shuffle, schema
variation, adversarial similar names, multilingual ID/EN, long tool lists.

Split by **template / value / phrasing**, not random row split.

Binary stream format (spec in `data/schema.md` when built): header magic +
version + n_samples + fixed layout per sample (token ids, masks, label ids).
Loader never maps whole file into process heap.

TDD: synthetic 8-sample fixture file; `next_batch` returns expected rows;
open does not allocate O(n_tokens_total).

---

## 7. THREE MODELS

| | A `01-sanity` | B `02-pilot-6m` | C `03-full-26m` |
|--|---------------|-----------------|-----------------|
| Goal | overfit 32–128 ex | real tool-calling | full arch quality |
| Termux B | 4 | 1 | 1 + act-ckpt + accum |
| Desktop B | 16+ | 4–8 | 2–4 |
| Peak est. | ~55 MB | ~180 MB | ~700 MB TIGHT |

Dirs: `models/01-sanity/`, `models/02-pilot-6m/`, `models/03-full-26m/`.

---

## 8. LOSS SCHEDULE

```text
L_total = L_generation [+ 0.1 L_retrieval] [+ λz L_z]
```

Weighted generation (stage 2+): punct 1.0×, tool name 2.0×, arg key 1.5×,
arg value 4.0×.

Stages: CE → weighted CE → retrieval → z-loss → INT4 QAT.
Baselines use CE only (Q10).

TDD: weighted CE on a 4-token fixture produces known scalar.

---

## 9. EVALUATION

**Headline metric (named once):** `val_full_call_EM`
(full function-call exact match on held-out; **higher is better**).

Also report:

* JSON parse rate
* tool-name exact match (`val_tool_name_EM`)
* argument-key precision/recall/F1
* argument-value exact match
* no-tool accuracy
* invalid tool rate / hallucinated arg rate
* runtime: prefill/decode tok/s, peak RSS, latency p50/p95

TDD: `eval/` fixture with hand-written preds/labels → exact metric numbers.

---

## 10. INFERENCE

* KV cache for decode
* greedy / top-k
* optional grammar-constrained JSON (stretch)
* streaming output (stretch)

TDD: cached vs uncached decode match on fixed prompt; cache memory bounded.

---

## 11. BUILD SYSTEM

Makefile first:

```text
make -C framework
make -C tests          # all unit/numerical — must be GREEN
make smoke MODEL=01-sanity
make train MODEL=01-sanity
make eval
```

Flags: `-std=c11 -O2 -Wall -Wextra`. Optional ASAN builds for tests.
CMake = stretch.

---

## 12. MILESTONES M0–M15 (GO/NO-GO)

| # | Milestone | GO / NO-GO |
|---|-----------|------------|
| M0 | Planning docs + skeleton | ✅ done |
| M1 | Tensor core | ✅ `tests/unit/test_tensor` green |
| M2 | Autograd min + free-graph | ✅ finite-diff; RSS flat |
| M3 | Linear + AdamW + CE toy | ✅ overfit toy |
| M4 | ZCRMSNorm / RoPE / GQA / gate | ✅ partial (residual path only); full attn bw stretch |
| M5 | Enc–dec + tied emb one step | ✅ integration test green |
| M6 | Stream dataloader + ckpt | ✅ round-trip; no full-load assert |
| M7 | Model A fixture smoke | ✅ 50 ep overfit synthetic id bins |
| **M8** | **BPE + synth FC data pipeline** | **⏳ CURRENT** — vocab 512/8192; C loader; JSONL → NDSET001 |
| **M9** | **Eval on real tool calls** | **⏳ CURRENT** — JSON parse + tool EM + full-call EM |
| **M10** | **Model A retrain on FC bins** | **⏳ CURRENT** — `val_full_call_EM > 0` on real val |
| **M11** | Model B pilot on FC | only after M10 keeps |
| **M12** | **Live `generate(query, tools)` + playground wired** | **⏳ CURRENT** — playground POST must call live model, NEVER hardcoded reference |
| M13 | Model C Termux path | ≥1 epoch no OOM or desktop-only note |
| M14 | Weighted CE / z-loss hooks | CE default; tests for weights |
| M15 | README comparison | wins/losses per model |

### Current focus (post-fixture)

Playground at `0.0.0.0:7860` currently prints a **hardcoded** Cactus reference
JSON — that is fake. M8–M12 are the current critical path:

```text
M8  BPE + synth FC corpus → NDSET001 stream bins
M9  eval JSON parse + tool-name EM + full-call EM (NOT just id token EM)
M10 Model A retrain on real FC bins (vocab 512 smoke / 8192 target)
M12 generate(query, tools) wired into playground — live, not stub
```

**Rule:** playground POST `/generate` returns model output. If the response is
the same regardless of `query`/`tools`, the build is broken — fix the stub.

GO before any B/C retrain, before any "experimentation loop" exercise:
playground produces input-dependent JSON.

Phases 0–8 map: oracle→M4/M9, tensor→M1, autograd→M2, attention→M4,
enc-dec+KV→M5/M12, sanity→M7/M10, pilot→M11, full→M13–14, infer-opt→M12+.

**No milestone advances if `make test` is red.**

---

## 13. WORKFLOW SUMMARY

See `program.txt`. Branch `needle.c/<tag>`. Experiment only under `models/`.
Maximize `val_full_call_EM`. Keep or `git reset --hard HEAD~1`. NEVER STOP
until human interrupts. Always run relevant tests before/after model edits
that touch shared assumptions; never break `make test`.

---

## 14. RISKS & MITIGATIONS

| Risk | Mitigation |
|------|------------|
| OOM / crash | HW §0.1; stream; free graph; B caps; act-ckpt C |
| Silent wrong attention | TDD oracle parity M4 before stack |
| Grad bugs | finite-diff every new op |
| Graph leak | RSS-flat smoke test |
| Metric hacking CE only | headline = full-call EM |
| Scope creep CUDA/FFN | out-of-scope list |
| Flaky data split | template-based split |
| Overfit eval | held-out only via eval/ |

---

## 15. REPRODUCIBILITY

* Seed all RNGs via `nd_seed(uint64_t)`.
* Log: seed, git hash, model config, batch, steps, peak RSS.
* Binary dataset version in header.
* Oracle dumps versioned under `tools/oracle/goldens/`.

---

## 16. VERIFICATION PLAN

```bash
make -C tests                 # unit + numerical must pass
make smoke MODEL=01-sanity    # overfit gate
make train MODEL=01-sanity
make eval
# RSS: grep peak_rss run.log ; must be <= 900 MB
```

---

## 17. SUCCESS CRITERIA

* Framework tests green (unit + grad + stream + RSS-flat).
* Model A overfits tiny set; full-call EM → high on that set.
* Model B non-trivial `val_full_call_EM` on held-out pilot data.
* Peak RSS ≤ 900 MB on all Termux train/eval paths.
* No full-file dataset load anywhere.
* Docs match code freeze zones.
* TDD discipline visible in git history (test commits before or with features).

---

## 18. CRITICAL FILES (creation order when coding starts)

1. `tests/unit/test_tensor.c` + `framework/include/tensor.h` (RED→GREEN)
2. `tests/unit/test_autograd.c` + autograd ops
3. `tests/numerical/test_gradcheck.c`
4. nn Linear, optim AdamW, CE + tests
5. `tests/numerical/test_attention_parity.c` + attention modules
6. enc–dec + `tests/integration/test_train_step.c`
7. stream dataset + tests
8. models/01-sanity train + smoke
9. eval harness + fixtures
10. pilot / full

---

## 19. FREEZE ZONES

| Zone | Rule |
|------|------|
| `framework/` | freeze after green tests; bugfixes only with new tests |
| `kernels/` | freeze after green |
| `tests/` | expand only; never weaken asserts to pass |
| `eval/` | fixed harness |
| `data/` artifacts | regen only with human approval |
| `models/*/` | editable in experiment loop |
| planning docs | human-owned |

---

## 20. OUT OF SCOPE (final)

CUDA-on-device, f16-for-RAM, epoch-cutting, full RAM load, default FFN stack,
distributed, HTTP server, external C deps v1, PyTorch runtime, CMake primary,
x86-first, edit frozen lib in loop, early INT4/BF16/Muon.

---

## 21. STRETCH

* ARM NEON kernels after CPU parity
* CUDA desktop backend
* BF16 train, Muon for projections, INT4 QAT
* grammar-constrained JSON decode
* contrastive retrieval head
* CMake multi-backend
* FFN ablation branch

---

END OF MASTER PLAN
