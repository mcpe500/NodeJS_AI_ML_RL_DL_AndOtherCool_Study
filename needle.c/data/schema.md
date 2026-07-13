# NDSET001 stream dataset schema

```
magic "NDSET001" (8 bytes)
version u32
n_samples u32
max_src u32
max_tgt u32
offsets[n_samples] u64   # file offsets only — never full corpus in RAM
records...
  src_len u32
  tgt_len u32
  src_ids[src_len] i32
  tgt_ids[tgt_len] i32
  labels[tgt_len] i32    # next-token teacher-forcing targets
```

API: `nd_dataset_open` → header+offsets; `nd_dataset_next_batch` → fseek+fread into caller buffers.
