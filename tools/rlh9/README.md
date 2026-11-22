# ENGINE 9 (rlh9.hpp): RLH8 superscalar codec, format R8F4

Build: `g++ -O3 -DNDEBUG -march=native -std=c++17 -DLLRICE_ENGINE=9 -DCODER=3 bench.cpp` (signature rev '9').
8-bit RGB only, width <= 16383. Design: Argon (rlh8f, split micro-FSM, dual tables, R8F3 geometry, 2026-11-21/22) on
Claude's RLH8s prototype; history in the lab journal 01c. Build tag: `llrice ENGINE=9 CODER=3 FMT=R8F4 RCT=rct`.

Format: llrice header, then per 16-row band: 4 x u32 lane byte sizes (bit 31 of the first = screen table set) + 4 lane
byte streams: Y rows 0..m-1, Y rows m..nb-1, S rows 0..m-1, S rows m..nb-1 (m = nb / 2).
Planes: RCT U = R-G, V = B-G, Y = G + ((U+V)>>2); MED per plane; zigzag. Chroma is one plane S = zU + zV (0..2040);
after every non-zero S token the lane carries zU in 0..S (zV = S - zU): for S <= 15 a static Huffman code (<= 6 bits)
per table set and S (RemLUT, Argon: the zigzag parity of S carries the sign agreement of eU, eV, so zU | S is far from
uniform at small S), for S >= 16 a truncated binary code.
Each band row of a plane is one chunk (EOB = rest of the row is zero). Tokens: 19 run classes x 44 quarter-octave value
classes + EOB (837).
Context per lane: split micro-FSM, st = sv | sr << 6 (sv: 48 states by the half-octave value class, calibrated on llrice
adapt_kr; sr: 16 states, EMA 1/4 of the run class, table listed literally). ctx (48 per group): photo set
(sv >> 1) * 2 + (sr >> 3), screen set (sv >> 2) * 4 + (sr >> 2). State (12, 0) at the start of every lane of every band.
Tables: 2 sets (photo, screen) x 2 groups (Y, S) x 48 = 192 static canonical Huffman tables (<= 15 bits),
plus 2 x 15 zU | S codes (RLH9_REMLEN).
The encoder takes the screen set when >= RLH9_ZTHR % (65) of the band's Y, U, V residuals are zero.
Decoder: entropy pass of a band with the 4 lanes interleaved one token per lane per round (packed 32-bit LUT, 9-bit
index, a class byte per LUT entry gives the next state, zU read branch-free from the same 64-bit window), then per
8 rows a wave MED in SSE2 (row k at column x - k; Argon, after the PLS wave decoder) and an SSSE3 inverse RCT
(pshufb to RGB24). -DRLH9_NO_SIMD or no SSE2: scalar, same output.

Tables: `rlh9_tables.h` from `tools/rlh9/rlh9_train.cpp` (1214 images, 1824 Mpx), Vladimir's lab split
(Google Drive: datasets/); the exact file names are in `tools/rlh9/train_list.txt`:
- photo set (`-f 0`): clic2020_professional_train, all 585;
- screen set (`-f 1`): train_gb82-sc_p6 (codec_wiki, gmessages, graph, gui; weight 4), screenshot_web without the 6 pages
  that are in screens_full12 (amazon, apple, cnn, creativecommons, imdb, microsoft, nytimes, reddit; weight 4),
  QOI suite screenshot_game (617 PNG; weight 1).
Test sets, never trained on: clic2020_professional_valid (41), screens_full12_p6 (12), Kodak.
Rebuild (reproduces the committed file byte for byte, ~2 min):
`tools/rlh9/train_tables.sh <clic train dir> <gb82 dir> <screenshot_web dir> <screenshot_game dir> [out.h]`
(each dir holds <stem>.ppm or <stem>.png). Lab tables vs the previous ones (536 CLIC, 4 gb82 + games):
valid41 8.7778 -> 8.7799 bpp, 12 screens 1.2640 -> 1.2557, Kodak 10.2199 -> 10.2196.

Measured (Claude's sandbox, Xeon ~3.8 GHz, 1 thread, -r 3):
| codec | test49 photos bpp / dec MB/s | 12 screens bpp / dec | web8 bpp / dec |
|---|---|---|---|
| PLS (plt2 palic-2r -s) | 8.8988 / 323 | 1.2914 / 661 | 1.9455 / 635 |
| E8 | 9.0331 / 86 | 1.3123 / 274 | 1.9497 / 217 |
| E9 R8F2 (8-row bands, Y/U/V lanes, fNBLI, scalar MED) | 8.9882 / 217 | 1.3154 / 524 | 1.9738 / 487 |
| E9 R8F3 (zU always truncated binary) | 9.0112 / 296 | 1.2670 / 942 | 1.9427 / 838 |
| E9 R8F4 (now, RemLUT for S <= 15) | 8.9734 / 284 | 1.2640 / 917 | 1.9370 / 827 |
| R8F3 with fNBLI colour | 9.0167 / 291 | 1.2720 / 937 | 1.9475 / 857 |
With S = zU + zV the fNBLI step V -= U>>2 no longer helps, so R8F3 uses the plain RCT (RLH9_RCT=0).
R8F3 lost +0.26% on photos vs R8F2; the RemLUT takes it back: R8F4 is -0.16% vs R8F2 on photos, -0.42% vs R8F3, for 3-4% of decode speed.
Why S works (Argon, chroma_mix reply): H(S) + H(zU|S) = H(zU, zV); U = R-G and V = B-G share -G, so the signs of eU, eV
agree in 69% of the photo pixels where both are non-zero, and the parity of S codes that for free; fNBLI and YCoCg-R
remove this correlation, max(zU, zV) and |eU| + |eV| + signs lose to the sum.
Checks: round trip of all three sets; 96 edge cases under ASan/UBSan with SIMD and with -DRLH9_NO_SIMD; 168 long-run
cases (runs up to 16382, widths 8200..16383); corrupt-stream fuzz 2 x 5000 mutations under ASan/UBSan; R8F2 and R8F3
files reject each other by the build tag. E0..E8 are not touched.
Encoder (2026-11-22, Argon's fast encoder, same output byte for byte): vector front end (AVX2 / SSSE3 / scalar,
cur[x-1] by register shift), nonzero bitmask + tzcnt in bands with >= 30% zeros, one branch-free 64-bit put per token,
tbin_tbl[2048] for the truncated code; ablation macros RLH9_ENC_MODE, RLH9_VENC_MODE, RLH9_RENC_MODE, RLH9_TBIN_MODE,
RLH9_FSM_ALU. Checked by Claude with separate binaries (old encoder from 01a4303 vs new AVX2 / SSE4.2 / -DRLH9_NO_SIMD,
cmp of the output files): 175 files (90 edge images 1x1..16383x17, valid41, 12 screens, web8, Kodak) x 3 builds, 0 diffs;
ASan/UBSan clean; the trainer gives identical tables. Encode MB/s, 1 thread: valid41 158 -> 299, 12 screens 421 -> 1700,
web8 369 -> 1195 (SSE build 279 / 1470 / 1048). Decoder unchanged.
Bug fixed by R8F3: the R8F1/R8F2 encoder put code + mantissa (up to 35 bits) into a 64-bit accumulator that could hold
31 pending bits, so a zero run >= 8192 with a long code (images wider than 8192) produced a corrupt stream.

Older measurements (R8F2 and R8F1):

R8F2 (8-row bands, Y/U/V lanes, fNBLI): see tools/rlh9/ablation/README.md.

R8F1, 49 held-out CLIC photos:
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
