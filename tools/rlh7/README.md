# RLH7 — two retrained RLH regimes (llrice engines 7 and 8)

`rlh7.hpp` is the ENGINE 6 (`rlh.hpp`) bitstream and state machine with **retrained static
probability tables**. The container, revision nibble layout, escape coding, mantissa packing and
the 32-token regime window are unchanged, so a decoder only needs the matching tables.

| engine | `RLH7_MODE` | tables | regimes × contexts | screens bpp | photos bpp | `sizeof(GlobalTables)` | init |
|--------|-------------|--------|--------------------|-------------|------------|------------------------|------|
| 6 (Argon shipped) | – | `rlh_tables.h` | 8 × 34 | 1.3604 | 8.8089 | 4 842 928 B | 1.76 ms |
| **7 fast** | 0 | `rlh7_tables_fast.h` + `rlh7_regime_fast.h` | 8 × 14 | **1.3454** | 8.8316 | **2 003 888 B** | **0.80 ms** |
| **8 strong** | 1 | `rlh7_tables_strong.h` + `rlh7_regime_strong.h` | 24 × 34 | **1.3123** | **8.7942** | 14 495 664 B | 7.77 ms |

`llrice.hpp` defines `RLH7_MODE` from `LLRICE_ENGINE`, so building with `-DLLRICE_ENGINE=7` or `8`
selects the pair automatically. Only `CODER=3` is implemented; any other value is a `#error`.

## Two changes to the model, not to the format

1. **The regime selector is trained, not hand-written.** ENGINE 6 maps the three block counters
   `(sum_rl, hi_vk, sum_ctx)` to a regime with fixed thresholds. Here a greedy decision tree
   (`learn_reg.py`) is fitted on block statistics dumped from the training corpus and emitted as a
   512-entry lookup table over `(min(sum_rl,256)>>4, min(hi_vk,15), min(sum_ctx,63)>>2)`.
2. **The smoothing weights are refitted and the domain prior is per regime.** The prior is
   `PRI[grp][reg]` (self-weight 4.0) instead of one vector per channel group, and the count scale is
   `4096 × {96, 24, 3.5}` by regime class. `--tune` re-fits the blend weights `α_c34 / α_c14 / α_reg`
   against the actual Huffman cost instead of the Dirichlet surrogate.

## Reproducing the tables

Training data: `gb82-sc` + `qoi-screenshot-web` minus the 12 `halic-wins-screen-p6` test images,
plus `icon-512`, `icon-64`, `screenshot-game`, and 20 CLIC training photos. Test sets are never used
for fitting.

```sh
# 1. token dumps (RLV1) from the training corpora
g++ -O2 -std=c++17 -I../.. tools/rlh7/dump_values.cpp -o dump
./dump <ppm_dir> <out.rlv>

# 2. block statistics for the selector, and the greedy tree
./block_stats ... ; python3 learn_reg.py --blocks ... -K 4000 --regs 24 --out mreg_24

# 3. tables (fast: 8 regimes, coarse 14-context blend, photo weight 6)
./train_rlh7 -o rlh7_tables_fast.h -r rlh7_regime_fast.h -lut mreg_8.lut \
             -nregs 8 -ctx 14 -ph 6 -tune
# strong: 24 regimes, full 34 contexts
./train_rlh7 -o rlh7_tables_strong.h -r rlh7_regime_strong.h -lut mreg_24.lut \
             -nregs 24 -ctx 34 -tune
```

`train_rlh7` caps the Huffman code length at 10 whenever the 11–12-bit codes would need more than
the 512 secondary decoder slots (`sub_need()` / `build_len()`). Without that cap the release build
silently corrupts the stream — ENGINE 6 never hit it (max 364 slots), the 24-regime tables do.

## Tests and measurement tools

* `test_fuzz_edge.cpp` — ASan+UBSan round-trips over 1×1…513×513, five patterns, 16-bit values and
  runs beyond 2^26. Both engines pass.
* `init_bench.cpp` — `sizeof(GlobalTables)` and best-of-5 construction time.
* `frontend_bench.cpp` — the pixel→residual→value front end with a null encoder.
  `-DFE_LEVEL=1` adds run-length tokenisation and regime bookkeeping, so the difference to the full
  encoder isolates the table-lookup + bit-writing cost.
