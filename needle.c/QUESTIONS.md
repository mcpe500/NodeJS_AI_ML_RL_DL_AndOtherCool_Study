# Questions encountered during execution

(Decisions made on the fly, with reasoning. Per user instruction: don't stop,
store questions here. Append-only.)

---

## Q0: Hardware-aware memory policy (2026-07-14)

Decision: HARD constraints codified in MASTER_PLAN §0.1, PLAN.md §0.1,
PLAN.txt, program.txt. Summary:

- Peak process RSS ≤ **900 MB**.
- Stream dataset rows; never full-load corpus into RAM.
- Free autograd graph every optimizer step.
- Model C Termux: B=1 + activation checkpointing + grad accum.
- Full float32 + full epochs kept.
- On OOM: log, halve batch once, then permanent `crash`.
- **Chunk / stream** data loads; never hold the full corpus in RAM.
- Longer wall-clock OK if it prevents crashes; still optimized (not needlessly long).

Reasoning: measured MemAvailable ~1.0–1.6 GB under load (often ~1.0 GB).
User reports frequent crashes. Design for MemAvailable, never MemTotal.
Full precision + full epochs stay; only load/activation strategy changes.
See PLAN.md §0.1.

---

## Q1: Repo layout + freeze zones

Decision:

- Frozen once green: `framework/`, `kernels/`, `eval/`, built `data/` artifacts.
- Editable in experiment loop: `models/{01-sanity,02-pilot-6m,03-full-26m}/` only.
- Planning docs (MASTER_PLAN etc.) are human-owned.

Reasoning: same pattern as token-predictor. Isolates experiment surface so
framework bugs don't masquerade as model wins. See program.txt CAN/CANNOT.

---

## Q2: C11 vs C99

Decision: **C11** (`-std=c11`).

Reasoning: `_Static_assert`, anonymous structs/unions, `stdalign.h` useful for
tensor/backend code. Termux clang supports C11. Public API still plain C.

---

## Q3: Makefile vs CMake

Decision: **Makefile first**. CMake is desktop stretch only.

Reasoning: Termux workflow is `make -C framework && make test`. Fewer deps,
matches token-predictor. CMake may be added later for multi-backend desktop
builds; never required on this device.

---

## Q4: Dataset reading — stream only

Decision: streaming only.

- `nd_dataset_open` reads header + builds sample index only.
- `nd_dataset_next_batch` does `fseek` + `fread` of exactly `batch` rows into
  caller-owned buffers.
- Never `malloc(n_samples * seq * …)`.
- Optional later: read-only `mmap` (kernel pages cold rows out). Not required.

Reasoning: full-load of function-calling token sequences + masks + labels
plus activations + AdamW state + autograd graph OOMs on ~1 GB available.
Full epochs + full float32 stay; only the *load strategy* changes.
Crashes from OOM are P0 bugs, not "try smaller data".

---

## Q5: Autograd graph free policy

Decision: free graph every optimizer step. Train loop contract:

```
zero_grad → forward → loss → backward → step → free(loss / preds / batch tensors)
```

`nd_tensor_free(loss)` must release the subgraph (refcount). No retention of
graphs across steps in the training loop.

Reasoning: graph leak was a crash source on token-predictor. Attention models
hold large score tensors (O(T²)); retaining them across steps blows the 900 MB
cap.

---

## Q6: Peak RSS enforcement

Decision: soft log of RSS when `/proc/self/status` is readable; hard path is
OOM (malloc NULL / kill). Target peak ≤ **900 MB**. Leave ~100–700 MB for OS
+ Termux + shell.

Reasoning: no reliable cgroup hard limit in Termux userland. Adaptive batch
(Q0) is the recovery path.

---

## Q7: Keep full precision and full epochs under RAM pressure?

Decision: **YES to both**. Never drop to f16/bf16 for RAM on the Termux path.
Never cut epoch count for RAM. Only reduce: batch size, seq length, enable
activation checkpointing, use grad accumulation for effective larger batch.
Streaming + graph free is the primary fix.

Reasoning: user explicit — full precision and full epochs matter; chunking
the *data load* and activation memory is the allowed optimization.

---

## Q8: Tied embedding

Decision: YES. Encoder input embedding, decoder input embedding, and output
projection share one `vocab × d_model` matrix.

Reasoning: matches Needle; cuts ~4M params on full model; stabilizes
generation. Framework must support weight tying (same storage, dual use).

---

## Q9: Gated residual init

Decision: initialize gate so residual starts near identity:

```
y = x + sigmoid(gate) * attention(norm(x))
```

with `gate` initialized to a large negative (or zero if using `sigmoid(gate)`
with zero-init convention documented in model code — prefer gate param such
that `sigmoid(gate) ≈ 0` at step 0). ZCRMSNorm `gamma` init to 0 so
`y ≈ x / rms` scales mildly.

Reasoning: Needle initializes gate and gamma near zero for stable early
training. Exact constant lives in model init; document in model README.

---

## Q10: Loss schedule

Decision: staged, never all at once:

1. plain cross-entropy
2. weighted CE (JSON punct 1.0×, tool name 2.0×, arg key 1.5×, arg value 4.0×)
3. contrastive tool-retrieval aux
4. z-loss
5. INT4 QAT (desktop / late)

Default for baselines: stage 1 only. Unlock later stages only after Model B
baseline is logged.

Reasoning: multi-loss from day one makes failure attribution impossible.

---

## Q11: Headline metric

Decision: **`val_full_call_EM`** — full function-call exact match on held-out
validation. Secondary: `val_tool_name_EM`, arg-value EM, JSON parse rate.
results.tsv primary column is `val_full_call_EM` (higher is better).

Reasoning: token CE alone can look healthy while JSON structure / tool choice
is wrong. Exact full-call match is the user-visible success criterion.

---

## Q12: No-FFN architecture

Decision: attention-only encoder–decoder. **No position-wise FFN/MLP** inside
layers. Capacity via depth, width, heads only. Gated residual + ZCRMSNorm +
RoPE + GQA are the building blocks.

Reasoning: matches Needle Simple Attention Network design. Ablation
"no-FFN vs FFN" is a stretch experiment, not the default stack.

---

## Q13: Test-Driven Development (TDD)

Decision: **TDD is HARD** for framework and numeric model contracts.

Order: write failing test → confirm RED → implement minimum → GREEN →
refactor → commit. Layers: `tests/unit`, `tests/numerical` (finite-diff +
oracle parity), `tests/smoke`, `tests/integration`, fixed `eval/` fixtures.

Rules:

* No feature without a test that would fail without it.
* Never weaken asserts to pass.
* `make -C tests` red → no milestone advance, no experiment `keep`.
* Model A smoke overfit is GO/NO-GO before Pilot/Full.
* RSS-flat N-step test required before multi-epoch default trains.

Reasoning: user request — less bug-prone. Attention/RoPE/mask bugs are silent
and expensive; tests are cheaper than 6M-param runs on a phone.
See MASTER_PLAN §0.2, PLAN.md §0.2, program.txt (TDD + HW rules).

---

## Q14: BPE vocab size for Termux real-FC train (2026-07-14)

Decision: **512 smoke / 8192 target**.

* Smoke path (default Termux first pass): vocab=512, max_src/tgt=64–128,
  Model A retrain, prove playground live.
* Target path: vocab=8192 once smoke GO; same pipeline, larger emb table.

Reasoning: tied emb size = V·d. At d=64, V=8192 → ~0.5M emb params alone —
fine for A, but BPE train + larger CE head slower on phone. 512 proves the
encode→train→generate path without waiting hours. Log which V was used in
`results.tsv` description.

---

## Q15: Playground must never hardcode tool JSON (2026-07-14)

Decision: **hardcoded reference JSON in playground is a bug**, not a feature.

POST `/generate` must call `generate(query, tools)` on live weights. If two
different queries produce byte-identical model output that matches a fixed
template independent of input, treat as fail. Showing a Cactus *example* in
docs is fine; injecting it as the response is not.

Reasoning: user typed `hello` and got San Francisco weather — false demo.
Trust requires input-dependent output even if underfit (garbled JSON OK;
hardcoded SF not OK).
