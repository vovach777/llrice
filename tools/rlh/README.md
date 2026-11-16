# RLH tools (Argon, 2026-11-16)

`LLRICE_ENGINE 6` = `rlh.hpp` + `rlh_tables.h` (static Huffman tables, 8 causal regimes switched every 32 tokens,
quarter-octave value classes, 34 contexts of the previous symbol, MAX_CLEN 12, decoder LUT 10 + 2 bits, rev '6').
- `train_rlh.cpp` — trains `rlh_tables.h` from value dumps of the llrice coders (`dumps/screens_train`,
  `dumps/clic_train20`, `dumps_c2/...`; dumps made with `rlhuff/dump.cpp` of plt2 `hlc/research/rlhuff`).
  Training sets: `data_screens_train_p6` (12 crops) + the first 20 CLIC 2020 train photos (without the one over 10 MB).
- `test_fuzz_edge.cpp` — ASan/UBSan stress: edge sizes, fills, a checkerboard with v = 2047, 16-bit streams, runs over
  2^26 zeros. `g++ -O1 -g -std=c++17 -march=native -mcrc32 -msse4.1 -fsanitize=address,undefined -DCODER=3
  -DLLRICE_ENGINE=6 test_fuzz_edge.cpp`.
