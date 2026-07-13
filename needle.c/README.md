# needle.c

PyTorch-like **C11** framework (`nd_*`) + **attention-only** encoder–decoder
for single-shot tool-calling JSON (Needle-style: no FFN, RoPE, ZCRMSNorm,
gated residual, GQA, tied emb).

**Status:** framework + A/B/C fixture train green; **real FC path (BPE + synth
JSONL + live generate)** in progress. Peak RSS ≤ 900 MB, f32, stream bins.

## HARDWARE AWARE (HARD — never violate)

| | |
|--|--|
| Host | Termux Android aarch64 (SM8350), 8 cores, **no GPU** |
| MemAvailable under load | **~1.0–1.6 GB** (design for this, not MemTotal ~7.1 GB) |
| Peak process RSS | ≤ **900 MB** |
| Precision | **full float32** — never drop for RAM |
| Epochs | **full configured count** — never cut for RAM |
| Dataset | **stream / chunk only** — never malloc whole corpus |
| Autograd | free graph every optimizer step |
| Model C | B=1 + grad accum 4 |
| OOM | log + permanent `crash` if >900 |
| Wall-clock | longer OK if it avoids crash |

**Method:** **TDD** — test first, then implement. See MASTER_PLAN §0.2.

## User try (Cactus-style) vs what exists

Official Needle lets a **user** type English + tool schema → tool-call JSON:

| | [Cactus Needle](https://cactuscompute.com/blog/needle#get-the-model) / [HF](https://huggingface.co/Cactus-Compute/needle) | **this repo (`needle.c`)** |
|--|--|--|
| UI | `needle playground` → http://127.0.0.1:7860 | `make playground` → 0.0.0.0:7860 |
| API | `generate(model, query, tools)` | `inference/generate.c` (BPE + greedy) |
| Tokenizer | BPE / shipped weights | `tokenizer/{vocab.json,merges.txt}` + `nd_bpe_*` |
| Weights | pretrained 26M on tool-call data | `make fc` → `ckpts/sanity_fc.nd` (vocab≈438) |
| Example in | natural language | query+tools JSON → model detok string |

**Cactus example (reference only — not runnable here yet):**

```text
INPUT  query:  What's the weather in San Francisco?
INPUT  tools:  [{"name":"get_weather","parameters":{"location":"string"}}]
OUTPUT:        [{"name":"get_weather","arguments":{"location":"San Francisco"}}]
```

```bash
# Cactus (other project) — real model + Gradio:
#   git clone https://github.com/cactus-compute/needle.git && cd needle && source ./setup
#   needle playground   # http://127.0.0.1:7860

# needle.c — real text→tool train then live playground:
python3 tokenizer/train_bpe.py                  # once
python3 data/synth/gen_fc.py                    # 2k/200/200 JSONL
python3 tools/jsonl_to_bin.py --jsonl data/raw/train.jsonl --out data/fc_train.bin
python3 tools/jsonl_to_bin.py --jsonl data/raw/val.jsonl   --out data/fc_val.bin
make fc                                         # Model A on FC bins → ckpts/sanity_fc.nd
make playground                                 # 0.0.0.0:7860 live generate()
# open http://127.0.0.1:7860/
make infer                                      # CLI generate(query,tools)
```

**Bind:** `0.0.0.0`. **Stop:** Ctrl+C.  
Output is **live model detok** (may underfit on small A). Not hardcoded.

---

## How to test (developer — works today)

From repo root `needle.c/`:

```bash
# 1) build + unit/integration tests
make -C framework
make -C tests
# expect last line: ALL TESTS GREEN

# 2) stream fixture bins (smoke 64 samples, small_train 256)
make -C models/01-sanity data

# 3) train Model A (overfit smoke, 50 epochs)
make smoke MODEL=01-sanity
# or: make train MODEL=01-sanity

# 4) eval checkpoint (metrics, not chat)
make -C eval
./eval/evaluate A data/smoke.bin models/01-sanity/ckpts/sanity.nd

# 5) real FC train (BPE vocab + fc_train.bin)
make fc
# then: make eval / make infer / make playground

# 6) B / C full epochs on fixture bins (still ≤900 MB peak)
make train MODEL=02-pilot-6m
make train MODEL=03-full-26m
```

Single-test targets:

```bash
make -C tests test_tensor
make -C tests test_autograd
make -C tests test_dataset
make -C tests test_train_step
```

## Example input / output (what actually runs)

### A) User chat (desired) — **not available**

See table above. Stub: `make chat-demo`.

### B) Fixture sample (what the model is trained on today)

Synthetic token-id sequences, vocab=64, lengths src=4 tgt=4.  
Writer: `tools/write_smoke_bin.c`. Sample index `i=0`:

| field | ids | meaning |
|-------|-----|---------|
| **src** (encoder) | `1 2 3 4` | prompt tokens |
| **tgt** (decoder teacher-force) | `2 3 4 5` | shifted input |
| **labels** (next-token) | `3 4 5 6` | targets for CE |

Rule: `a = (i + j + 1) % 64` → `src=a`, `tgt=a+1`, `label=a+2`.  
Packed as streamable `NDSET001` (`data/schema.md`).

### C) Train log (Model A smoke)

```text
$ make smoke MODEL=01-sanity
mode=smoke n_params=78278 vocab=64 d=64
smoke epoch=1/50 loss=3.2456 steps=16 peak_rss_mb=4.4 cur_rss_mb=4.4
smoke epoch=50/50 loss=0.6756 steps=16 peak_rss_mb=4.5 cur_rss_mb=4.4
smoke first_loss=3.6016 last_loss=0.6220
token_EM=0.0195 val_full_call_EM=0.0000 peak_rss_mb=4.5
peak_rss_mb=4.5 status=keep
```

OK if: loss↓, `peak_rss_mb` flat, `status=keep`, peak ≤ 900.

### D) Eval (metrics)

```text
$ ./eval/evaluate A data/smoke.bin models/01-sanity/ckpts/sanity.nd
model=A
token_EM=0.1562
val_full_call_EM=0.0000
peak_rss_mb=3.8
```

| field | meaning |
|-------|---------|
| `token_EM` | argmax(logits)==label rate |
| `val_full_call_EM` | all positions correct (headline) |
| `peak_rss_mb` | VmHWM ≤ 900 |

### E) Infer (token ids only)

```text
$ make infer
greedy_ids: 2 0 0 0
peak_rss_mb=3.1
{"tool":"demo","args":{"ids":[2,0,0,0]}}
```

Fixed toy ids in → argmax ids out. JSON shell is **demo packaging**, not a real tool call.

### F) Unit test

```text
$ make -C tests test_train_step
n_params=78278
loss0=4.1543 loss1=3.8770 rss=3.7
test_train_step: OK
```

## Results (fixture vocab=64, stream bins)

| Model | params | epochs | last loss | token_EM | val_full_call_EM | peak_rss_mb | status |
|-------|--------|--------|-----------|----------|------------------|-------------|--------|
| A 01-sanity smoke | 78k | 50 | 0.62 | 0.02 | 0.00 | **4.5** | keep |
| A 01-sanity train | 78k | 20 | 0.55 | 0.16 | 0.00 | **4.6** | keep |
| B 02-pilot-6m smoke | 4.0M | 5 | 0.96 | 0.00 | 0.00 | **65** | keep |
| B 02-pilot-6m train | 4.0M | 10 | 0.44 | 0.58 | 0.03 | **65** | keep |
| C 03-full-26m smoke | 22M | 2 | 2.62 | 0.00 | 0.00 | **346** | keep |
| C 03-full-26m train | 22M | 3 | 1.00 | 0.00 | 0.00 | **346** | keep |

Headline metric: **`val_full_call_EM`** (higher better). Fixture is synthetic id sequences — not real tool JSON yet; BPE + synth FC data is next polish.

## Docs

1. [`program.txt`](program.txt) — agent loop / rules
2. [`PLAN.md`](PLAN.md) — condensed reference (+ §0.1 HW)
3. [`MASTER_PLAN.md`](MASTER_PLAN.md) — definitive SoT
4. [`QUESTIONS.md`](QUESTIONS.md) — decisions (Q0 HW)
5. [`PLAN.txt`](PLAN.txt) — narrative genesis
6. [`data/schema.md`](data/schema.md) — NDSET001 stream format
