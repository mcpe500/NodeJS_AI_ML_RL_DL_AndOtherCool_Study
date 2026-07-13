# needle.c PLAN

Reference plan for the experiment defined in `program.txt`. Architecture,
framework API, dataset, milestones, **HARDWARE AWARE** caps, **TDD** rules.
Read once at setup; refer when generating experimental ideas.

## 0. Goal in one paragraph

Build a PyTorch-like C11 framework (`nd_*`) and an attention-only
encoder–decoder tool-calling model (Needle-style: no FFN, RoPE, ZCRMSNorm,
gated residual, GQA, tied emb) at three sizes (sanity / pilot 6M / full 26M).
Train and eval on streamed function-calling data. Headline metric:
**`val_full_call_EM`**. Method: **TDD** — test first, then code.

## 0.1 HARDWARE AWARE (HARD CONSTRAINTS — never violate)

**Measured host (Termux Android aarch64, 2026-07-14):**

| Resource | Value |
|----------|-------|
| Platform | Termux Android aarch64, Qualcomm SM8350 (Venus) |
| CPU | 8 cores, **NO GPU** |
| MemTotal | ~7.1 GB |
| MemAvailable (under load) | **~1.0–1.6 GB** (often ~1.0 GB) |
| Swap | 4 GB (often half used) |

**Rule: design for MemAvailable, never MemTotal.**

### Hard caps

| Cap | Value | Why |
|-----|-------|-----|
| Precision | **float32 only** (full) | Spec. No f16/bf16 for RAM. |
| Epochs | **full epoch count** | Never skip epochs to "save RAM". |
| Dataset load | **streaming only** — never malloc whole corpus | Activations + AdamW + graph already heavy. |
| Peak RSS | ≤ **900 MB** | Leave headroom for OS/Termux. |
| Autograd | **free every step** after `opt->step` | Graph leak → crash. |
| Model A batch | def 4, max 8, min 1 | Tiny model. |
| Model B batch | def 1, max 2, min 1 | ~6M + T=256. |
| Model C batch | **1** + act-ckpt + grad accum 4–8 | ~700 MB peak TIGHT. |
| Parallelism | single-thread first | No N workers × batch. |
| OOM policy | log + halve batch once → permanent `crash` | Don't thrash. |

### Streaming dataset contract (required)

**Chunking rule (user requirement):** training may take longer if needed, but
must not crash. Prefer many small streamed batches over one giant in-RAM load.
Full float32 + full epochs stay; only the *load/activation* strategy absorbs
RAM pressure. Do not make runs needlessly long — stream + free-graph is enough
for A/B; C adds act-ckpt + grad accum.


```
nd_dataset_open(path)   → FILE* + header + index only
nd_dataset_next_batch() → fseek + fread ONLY requested rows into caller buffers
nd_dataset_close()      → fclose + free indices
```

- **Forbidden:** load entire corpus into one giant buffer.
- **Allowed later:** read-only mmap. Prefer fseek first.

### Adaptive batch (anti-crash)

On OOM (`malloc` NULL / killed):

1. Log `crash` + current batch.
2. Halve batch (floor 1).
3. Retry once.
4. Still OOM → permanent crash; abandon that idea.

### Memory budget (float32 rough)

| Model | params | peak |
|-------|--------|------|
| A Sanity d=64 L=2/2 T=64 B=4 | ~107k | **~55 MB** |
| B Pilot d=256 L=8/6 T=256 B=1 | ~6.0M | **~180 MB** |
| C Full d=512 L=12/8 T=512 B=1 | ~26.2M | **~700 MB TIGHT** |

All under 900 MB **iff** stream + free graph (+ act-ckpt on C).

### What stays full

- Full float32. Full epochs. Full tier capacity.
- Full streamed eval.
- RAM levers only: stream, free graph, microbatch/accum, checkpoint, lower B/T.

## 0.2 TDD (HARD — never skip)

```text
RED → GREEN → refactor → commit
```

| Rule | Detail |
|------|--------|
| Test before feature | No op/module/metric without a failing test first |
| Numerical parity | vs Python oracle for attention block |
| Grad check | finite-diff every new backward |
| RSS-flat | N-step loop must not leak |
| Smoke gate | Model A overfit before Pilot/Full |
| `make test` red | automatic revert / no milestone advance |
| Frozen tests | never weaken asserts to pass |

Layers: `tests/unit`, `tests/numerical`, `tests/smoke`, `tests/integration`,
fixed `eval/` fixtures. See MASTER_PLAN §0.2.

## 1. Directory layout

```
needle.c/
├── program.txt                 ← ops loop (read first for agents)
├── PLAN.md                    ← this file
├── MASTER_PLAN.md             ← full SoT
├── PLAN.txt                   ← verbose genesis
├── QUESTIONS.md               ← ADR log
├── Makefile
├── framework/                 ← FREEZE once green
├── kernels/                   ← FREEZE once green
├── tests/                     ← TDD home; expand only
├── models/                    ← EDIT in experiment loop
│   ├── 01-sanity/
│   ├── 02-pilot-6m/
│   └── 03-full-26m/
├── tokenizer/  training/  inference/  tools/
├── data/                      ← FREEZE artifacts once built
├── eval/                      ← FREEZE harness
└── scripts/
```

## 2. Framework API (contracts only)

```c
typedef struct nd_tensor {
    float *data, *grad;
    int64_t shape[8], strides[8];
    int32_t ndim, size;
    bool requires_grad, owns_data;
    struct nd_node *grad_fn;
    int refcount;
} nd_tensor;

nd_tensor *nd_zeros(...);
void       nd_tensor_free(nd_tensor *t);  /* free subgraph */
nd_tensor *nd_matmul(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_cross_entropy(nd_tensor *logits, nd_tensor *targets);
void       nd_backward(nd_tensor *loss);

nd_dataset *nd_dataset_open(const char *path);  /* stream */
int         nd_dataset_next_batch(nd_dataset *d, ..., int batch);

nd_optim *nd_adamw_create(nd_module *m, float lr, float wd);
void      nd_optim_step(nd_optim *o);

/* arch building blocks */
nd_module *nd_zcrmsnorm(int d, float eps);
nd_module *nd_rope_gqa(int d, int n_q, int n_kv, bool causal);
nd_module *nd_encoder_stack(...);
nd_module *nd_decoder_stack(...);
nd_module *nd_needle_create(const nd_needle_config *cfg);
```

Train free contract:

```
zero_grad → forward → loss → backward → step → free(loss/preds)
```

## 3. Model A / B / C

| | A `01-sanity` | B `02-pilot-6m` | C `03-full-26m` |
|--|---------------|-----------------|-----------------|
| d / Lenc/Ldec / T | 64 / 2/2 / 64 | 256 / 8/6 / 256 | 512 / 12/8 / 512 |
| heads/kv | 4/2 | 8/4 | 8/4 |
| vocab | 8192 | 8192 | 8192 |
| Termux B | 4 | 1 | 1 + act-ckpt + accum |
| Desktop B | 16+ | 4–8 | 2–4 |
| Goal | overfit | real FC | full quality |
| Peak | ~55 MB | ~180 MB | ~700 MB |

Shared: enc–dec, **no FFN**, RoPE, ZCRMSNorm, gated residual, tied emb, GQA.

```text
ZCRMSNorm: y = x * (1+gamma) / sqrt(mean(x²)+eps)
Gated:     y = x + sigmoid(gate) * attn(norm(x))
```

## 4. Loss, eval, headline metric

**Headline:** `val_full_call_EM` (higher better).

Secondary: `val_tool_name_EM`, arg-value EM, JSON parse rate, no-tool acc,
peak RSS, tok/s.

Loss stages: CE → weighted CE → retrieval → z-loss → INT4 QAT.
Baselines = CE only.

## 5. Build targets

```text
make -C framework
make -C tests          # must be GREEN
make smoke MODEL=01-sanity
make train MODEL=<id>
make eval
```

## 6. Milestones M0–M15 (one-liners)

| # | Gate |
|---|------|
| M0 | docs + dirs; HW+TDD present |
| M1 | test_tensor RED→GREEN |
| M2 | autograd + RSS-flat |
| M3 | Linear+AdamW+CE overfit 10 |
| M4 | attention oracle parity |
| M5 | enc–dec one train step |
| M6 | stream loader + ckpt |
| M7 | Model A smoke; RSS≤900 |
| M8 | BPE + data bins |
| M9 | eval fixtures green |
| M10 | A baseline results.tsv |
| M11 | B pilot baseline |
| M12 | KV-cache JSON decode |
| M13 | C Termux 1 epoch or desktop-only |
| M14 | weighted CE hooks |
| M15 | README comparison |

**No advance if `make test` red.**

## 7. Risks

| Risk | Mitigation |
|------|------------|
| OOM | §0.1 stream/free/B caps/ckpt |
| Wrong attn | TDD oracle M4 |
| Grad bugs | finite-diff |
| CE-only illusion | headline full-call EM |
| Scope creep | OOS list |

## 8. Reproducibility

Seed via `nd_seed`. Log git hash, config, batch, peak RSS. Dataset header version.
Oracle goldens versioned.

## 9. Out of scope

CUDA-on-device, f16-for-RAM, cut epochs, full RAM load, default FFN, distributed,
HTTP server, C deps v1, PyTorch runtime, CMake primary, x86-first, edit frozen
lib in loop, early INT4/BF16/Muon.

## 10. Success criteria

* Tests green (unit + numerical + smoke + integration).
* A overfits; B non-trivial `val_full_call_EM`.
* Peak RSS ≤ 900 MB all Termux paths.
* No full-file load.
* TDD visible in history.
