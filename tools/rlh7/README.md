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

## Layout research: cross-plane sub-streams (XPL), 2026-11-18

Model tools only; the engine is unchanged.

* `cross_model.cpp` (Claude) — planar layout with one coder, where plane r is split into sub-streams by a class
  of g at the same pixel and plane b by a class of (g, r). Runs are counted inside a sub-stream; each sub-stream
  has its own tables x ctx34 and its own EOB per chunk. Per-image semi-static Huffman + header estimate.
  `XPL0` is a self-check: it equals SEP bit for bit.
* `xpl_model_ext.cpp` (Hydrogen) — extends it with b contexts from the same-pixel r magnitude:
  XPL4 sig(g) / rmag4(r), XPL5 rmag5(r), XPL6 sig(g) x rmag4(r), XPL7 lv3(r).
* `xpl_roundtrip.cpp` (Hydrogen) — static canonical Huffman tables (length cap 15), 2-fold CV over the
  screen and photo sets together, real bit-level encode + decode of every image, `mismatches` must be 0.
  One global table set (no regime selector), so absolute bpp is above the engine's.

```sh
g++ -O3 -march=native -std=c++17 -I../.. tools/rlh7/xpl_model_ext.cpp -o xpl_model_ext
./xpl_model_ext <ppm_dir>
g++ -O3 -march=native -std=c++17 -I../.. tools/rlh7/xpl_roundtrip.cpp -o xpl_roundtrip
./xpl_roundtrip <screens_ppm_dir> <photos_ppm_dir>
```

Reference (12 `halic-wins-screen-p6` / 30 CLIC, C512, EOB, exact runs 0..7), bpp:

| layout | screens, per-image | CLIC30, per-image | screens, static CV | CLIC30, static CV |
|---|---|---|---|---|
| SEP  | 1.2673 | 8.7742 | 1.4724 | 9.1701 |
| ONE ph3 | 1.1385 | 8.8117 | 1.4636 | 9.3908 |
| GRB ph2 | 1.2016 | 8.7357 | 1.4314 | 9.1665 |
| XPL2 | 1.1241 | 8.8407 | 1.2837 | 9.1899 |
| XPL4 | 1.1450 | 8.7636 | 1.3172 | 9.1294 |
| XPL6 | 1.1233 | 8.7873 | 1.2861 | 9.1480 |

CLIC30 = `sorted(585 CLIC train .plr)[0::20][:30]`, 66 322 840 px (reconstruction of the set used for
`chunk_model`; same pixel total). Screens and the per-image columns were re-run by Claude and match to 4 digits.
