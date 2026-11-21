# ENGINE 9 (rlh9.hpp): RLH8 superscalar photo codec, format R8F1

Build: `g++ -O3 -DNDEBUG -march=native -std=c++17 -DLLRICE_ENGINE=9 -DCODER=3 bench.cpp` (signature rev '9').
8-bit RGB only, width <= 16383. Design: Argon (rlh8f, 2026-11-21) on Claude's RLH8s prototype; history in the lab journal 01c.

Format: llrice header (16 B), then per 8-row band: 3 x u32 lane byte sizes + lane Y, U, V byte streams.
Each band row of each plane is one chunk (EOB = rest of the row is zero). Tokens: 19 run classes x 44 quarter-octave value
classes + EOB (837). Context per lane: ctx = min(kr >> 2, 15) * 3 + min(previous run, 2), kr = llrice RLGR value-k state,
carried across rows and bands; 2 table groups (Y, shared U/V) = 96 static canonical Huffman tables (<= 15 bits).
Decoder: entropy pass of a band with the 3 lanes interleaved one token per lane per round (packed 32-bit LUT, 9-bit index,
one load per token, branchless adapt), then fused unzigzag + branchless MED + inverse RCT.

Tables: `rlh9_tables.h` from `tools/rlh9/rlh9_train.cpp`: CLIC 2020 professional train, 536 of 585 photos
(the 49 held out: sorted clic-palic2 color/*.pln, index % 12 == 3), 1246 Mpx:
`DEC=palic-2 SKIP=test49.txt ./rlh9_train rlh9_tables.h <clic-palic2>/color`

Measured (Claude's sandbox, Xeon ~3.8 GHz, -r 3, 49 held-out CLIC photos):
| engine | bpp | enc MB/s | dec MB/s |
|---|---|---|---|
| E7 | 9.0704 | 162 | 88.5 |
| E8 | 9.0331 | 161 | 82.7 |
| E9 (old RCT, e1ddfed) | 9.0006 | 152 | 177 |
| E9 (fNBLI colour, R8F1 now) | 8.9807 | 150 | 172 |
Colour: the fNBLI / PALIC transform (Y = G + ((U+V+2)>>2), then V -= U>>2; V stays -255..255): -0.22% on photos.
The same transform for the RLGR / RLH7 path is the option LLRICE_RCT_FNBLI (default 0): photos E1 -0.20%,
E7 -0.24%, E8 -0.24%; 12 screens +0.4% worse.
Screens are not a target: 12 classic screens E8 1.3123, E9 1.4647 bpp (photo-only tables, no regimes).
Checks: round trip 49 photos + 33 edge sizes ok; E0..E8 encoder output byte-identical before/after (54 files);
corrupt-stream fuzz under ASan/UBSan (10k mutations): decoder throws or decodes, no memory errors.
