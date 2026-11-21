# ENGINE 9 (rlh9.hpp): RLH8 superscalar codec, format R8F2

Build: `g++ -O3 -DNDEBUG -march=native -std=c++17 -DLLRICE_ENGINE=9 -DCODER=3 bench.cpp` (signature rev '9').
8-bit RGB only, width <= 16383. Design: Argon (rlh8f + split micro-FSM and dual tables, 2026-11-21/22) on Claude's
RLH8s prototype; history in the lab journal 01c. Build tag: `llrice ENGINE=9 CODER=3 FMT=R8F2 RCT=fnbli`.

Format: llrice header, then per 8-row band: 3 x u32 lane byte sizes (bit 31 of the first = screen table set)
+ lane Y, U, V byte streams. Each band row of each plane is one chunk (EOB = rest of the row is zero). Tokens: 19 run
classes x 44 quarter-octave value classes + EOB (837).
Context per lane: split micro-FSM, st = sv | sr << 6. sv: 48 states driven by the half-octave value class of the token
(calibrated on llrice adapt_kr), sr: 16 states, EMA 1/4 of the run class (table listed literally in rlh9.hpp).
Both depend on the Huffman symbol only. ctx (48 per group): photo set (sv >> 1) * 2 + (sr >> 3),
screen set (sv >> 2) * 4 + (sr >> 2). Start state sv 12, sr 0, carried across rows and bands.
Tables: 2 sets (photo, screen) x 2 groups (Y, shared U/V) x 48 = 192 static canonical Huffman tables (<= 15 bits).
The encoder takes the screen set when >= RLH9_ZTHR % (65) of the band residuals are zero.
Decoder: entropy pass of a band with the 3 lanes interleaved one token per lane per round (packed 32-bit LUT, 9-bit index,
plus a class byte per LUT entry that gives the next state), then fused unzigzag + branchless MED + inverse RCT.

Tables: `rlh9_tables.h` from `tools/rlh9/rlh9_train.cpp` (1157 images, 1648 Mpx):
- photo set (`-f 0`): CLIC 2020 professional train, 536 of 585 photos (the 49 held out: sorted clic-palic2
  color/*.pln, index % 12 == 3);
- screen set (`-f 1`): gb82 screenshots codec_wiki, gmessages, graph, gui (weight 4) + QOI suite screenshot_game
  (617 PNG, one unreadable file skipped). Not trained on: the 12 classic screens (halic-wins-screen-p6) and web8
  (the 8 tall QOI web pages), both are test sets.
`SKIP=test49.txt PNG2PPM="python3 png2ppm.py" ./rlh9_train rlh9_tables.h -f 0 <clic ppm dir> -f 1 -w 4 <gb4 dir> -w 1 <qoi game dir>`

Measured (Claude's sandbox, Xeon ~3.8 GHz, -r 3):
| engine | test49 photos bpp / dec MB/s | 12 screens bpp / dec | web8 bpp / dec |
|---|---|---|---|
| E8 | 9.0331 / 86 | 1.3123 / 274 | 1.9497 / 217 |
| E9 R8F1 (kr x run ctx, photo tables) | 8.9807 / 173 | 1.4647 / 462 | 2.0121 / 422 |
| E9 R8F2 (now) | 8.9882 / 217 | 1.3154 / 524 | 1.9738 / 487 |
Photo cost of the FSM context: +0.08% (the screen-style ctx split costs +0.28% on photos, hence a split per set).
Detector threshold 50..72 changes screens by < 0.4%; 65 keeps photos best.
Checks: round trip of all three sets, 96 edge cases (24 sizes x 4 patterns, incl. alternating flat / noise bands) under
ASan/UBSan; corrupt-stream fuzz 2 x 5000 mutations (photo and screen crops) under ASan/UBSan: throws or decodes, no errors;
an R8F1 file is rejected by the R8F2 decoder by the build tag and the other way round. E0..E8 are not touched.

Older measurements (R8F1, 49 held-out CLIC photos):
| engine | bpp | enc MB/s | dec MB/s |
|---|---|---|---|
| E7 | 9.0704 | 162 | 88.5 |
| E8 | 9.0331 | 161 | 82.7 |
| E9 (old RCT, e1ddfed) | 9.0006 | 152 | 177 |
| E9 (fNBLI colour, R8F1) | 8.9807 | 150 | 172 |
Colour: the fNBLI / PALIC transform (Y = G + ((U+V+2)>>2), then V -= U>>2; V stays -255..255): -0.22% on photos.
The same transform for the RLGR / RLH7 path is the option LLRICE_RCT_FNBLI (default 0): photos E1 -0.20%,
E7 -0.24%, E8 -0.24%; 12 screens +0.4% worse.
Checks: round trip 49 photos + 33 edge sizes ok; E0..E8 encoder output byte-identical before/after (54 files);
corrupt-stream fuzz under ASan/UBSan (10k mutations): decoder throws or decodes, no memory errors.
