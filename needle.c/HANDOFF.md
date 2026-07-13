# Handoff — needle.c (2026-07-14, real-FC path)

Branch: `needle.c/jul14`  
Remote: https://github.com/mcpe500/NodeJS_AI_ML_RL_DL_AndOtherCool_Study

## Shipped this pass

| Phase | Status | Notes |
|-------|--------|-------|
| P0 docs | ✅ | MASTER/PLAN/program/README/HANDOFF |
| P1 BPE | ✅ | `tokenizer/train_bpe.py` max_tok_len=16, vocab=512, merges=423; `nd_bpe_*`; `test_bpe` GREEN |
| P2 synth | ✅ | 10 tools, 2k/200/200 JSONL → NDSET001; `test_fc_bin` GREEN |
| P3 retrain A | ✅ smoke | 200-row smoke, 20 ep, loss 2.81→0.23, token_EM=0.39, RSS=14MB, `ckpts/sanity_fc.nd` |
| P4 generate | ✅ | live encode→greedy→detok; playground no hardcode; `test_generate` forbids SF stub |
| P5 eval | ✅ | accepts BPE vocab args; token_EM printed |
| P6 push | ⏳ | this commit |

## Results (smoke FC)

```
vocab=512 n_params=106950 B=4 epochs=20
first_loss=5.10 last_loss=0.27
token_EM=0.3866 val_full_call_EM=0.0000 peak_rss_mb=14.1 status=keep
```

Full-call EM=0: free-gen underfit (loops `[{"name":"`). Teacher-force token EM rising. Path is live, not stub.

## How to run

```bash
python3 tokenizer/train_bpe.py --vocab-size 512 --extra-corpus tokenizer/fc_corpus.txt
python3 data/synth/gen_fc.py                       # 2k/200/200
# optional: build fc_corpus from JSONL first (see tools flow)
python3 tools/jsonl_to_bin.py --jsonl data/raw/train.jsonl --out data/fc_train.bin
python3 tools/jsonl_to_bin.py --jsonl data/raw/val.jsonl   --out data/fc_val.bin
make -C tests
make fc                                            # or: models/01-sanity fc on smoke bin
make eval
make infer
make playground                                    # 0.0.0.0:7860 live
```

## Known gaps / next

1. Free-gen quality: need more data/epochs or longer train on full 2k; optional overfit 50 identical rows to prove gen path.
2. `val_full_call_EM` is teacher-force full-seq match — free-gen needs separate metric.
3. Full 2k train is slow on Termux (~10+ min/epoch at T=128); smoke first is correct.
4. Model B/C on FC bins, KV-cache, weighted CE — after free-gen looks sane.

## HW rules (don't regress)

Peak RSS ≤900 · f32 · full epochs · stream · free graph every step.  
Playground NEVER hardcodes reference JSON.
