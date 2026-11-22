# ENGINE 9 (rlh9.hpp): RLH8 superscalar codec, format R8F5 (R8F4 below is history, commit 1c440de)

Build: `g++ -O3 -DNDEBUG -march=native -std=c++17 -DLLRICE_ENGINE=9 -DCODER=3 bench.cpp` (signature rev '9').
8-bit RGB only, width <= 16383. Design: Argon (rlh8f, split micro-FSM, dual tables, R8F3 geometry, 2026-11-21/22) on
Claude's RLH8s prototype; history in the lab journal 01c. Build tag: `llrice ENGINE=9 CODER=3 FMT=R8F5 RCT=rct` (R8F4 files are rejected).


## R8F5 (2026-11-23): learned 48-state context FSM, state = context

Same bands, lanes, RCT, MED, alphabet and RemLUT as R8F4 below. Only the context machine changes. R8F4 keeps 768 hidden
states (sv 48 x sr 16) and projects them to 48 contexts per group; R8F5 has exactly 48 states per group (Y, S) and set
(photo, screen), and the state IS the context: next = R8F5_TRANS[set][group][ctx][cls], cls = (rc << 4) | vh as in R8F4.
The decoder stores the next context in each LUT entry (lut_next, same size as the old class byte), so a token costs one
dependent load: peek -> lut[(tc << LUTB) | bits] -> tc'. Lanes start at R8F5_INIT_CTX; EOB keeps the context.
Tables: r8f5_best48_tables.h (R8F5_TRANS + R8F5_LEN[192][419]); RLH9_REMLEN still comes from rlh9_tables.h.

T and the 192 Huffman tables are learned together by EM (cross-entropy k-means): E-step picks for each (ctx, cls) the
next ctx minimising the cost of the following symbols under the current tables (min weight 25), M-step rebuilds the
tables. Design and prototype: Argon (tools/rlh9/r8f5_exp.cpp, in-RAM, 2026-11-22). Full-corpus training: WorkBuddy,
tools/rlh9/r8f5_stream.cpp (streaming counters, 1.5 GB RAM, 11 min on 4 cores): tools/rlh9/train_list.txt with the
train_tables.sh weights, 10% of every group held out (every 10th file by name), 8 EM iterations, iteration chosen by the
held-out bpp (8th). Checked by Claude: encoders AVX2 / SSE4.2 / -DRLH9_NO_SIMD write the same bytes and all three
decoders return the source on 175 files (edge sizes, valid41, 12 screens, web8, Kodak); fuzz under ASan/UBSan
(tools/rlh9/fuzz_r8f5.cpp 2 x 5000 per build, plus 500 mutations on each of 90 edge images) clean.

| bpp / dec MB/s, 1 thread AVX2 | R8F4 (1c440de) | R8F5 |
|---|---|---|
| valid41 | 8.7799 / 308 | 8.7446 (-0.40%) / 345 |
| Kodak | 10.2196 / 306 | 10.1921 (-0.27%) / 311-354 |
| 12 screens | 1.2557 / 1263 | 1.2682 (+1.00%) / 1320 |
| web8 (train) | 1.9304 / 1108 | 1.9015 (-1.50%) / 1200 |
| 6 private iPhone screens (Claude only) | 2.0206 | 1.9305 (-4.5%) |

On the 12 screens R8F5 is better or equal on 7 images; the loss is phoboslab (+3.3%) and windows95 (+2.3%): the screen
set learned mostly from the 617 QOI game screenshots. Open: screen set retrained with more web/phone/scan screenshots
and per-set iteration choice (photo and screen tables are independent and can be replaced separately).
Not taken: 2-token LUT (RLH9_R8F5_2TOK, slower on AVX2 photos); R8F4 "part A" (pre-projected FSM tables, same format):
no gain on Claude's machine. tools/rlh9/verify_r8f5_bin.cpp is Argon's two-binary verifier (needs local copies
rlh9_1c440de_orig.hpp and rlh9_r8f5.hpp).

## R8F4 (history)

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
web8 369 -> 1195 (SSE build 279 / 1470 / 1048).
Decoder (2026-11-22, Argon's fast decoder, same pixels): slow path (codes > 9 bits) by a 2-level O(1) table sub_lut
instead of the noinline decode_slow loop (the call made gcc spill the 4 lane registers in the hot loop); 2-lane drains
Y0+Y1 / S0+S1 before the 1-lane tails; tbin_tbl[2048] for the truncated code; MED wave in 3 full AVX2 passes per 16-row
band (Y0-7|U0-7, V0-7|Y8-15, U8-15|V8-15) and iRCT AVX2 by 32 px; SSE2/SSSE3 and scalar paths kept. Ablation macros
RLH9_DEC_NO_MEMSET, RLH9_DEC_2L_DRAIN, RLH9_DEC_LUT64, RLH9_DEC_SLOW_MODE, RLH9_DEC_TBIN_MODE, RLH9_DEC_WAVE_MODE,
RLH9_DEC_IRCT_AVX2 (lut64 is built only with RLH9_DEC_LUT64=1). Checked by Claude with separate binaries (4ad7ef9
encoder, old decoder vs new AVX2 / SSE4.2 / -DRLH9_NO_SIMD / LUT64=1, cmp): 175 files x 4 builds, 0 diffs; fuzz under
ASan/UBSan (tools/rlh9/fuzz_dec.cpp 2 x 5000, plus 1000 mutations on each of 92 edge images and crops), no errors,
same accepted/rejected streams as the old decoder. tools/rlh9/dec_verify_bin.cpp: Argon's two-binary verifier.
Decode MB/s, 1 thread, old -> new: valid41 288 -> 310, 12 screens 970 -> 1300, web8 882 -> 1143, Kodak 262 -> 307
(SSE build 256 -> 278, 918 -> 1064, 814 -> 923, 224 -> 243). Cumulative ablation, AVX2, valid41 / screens / web8:
| step | valid41 | screens | web8 |
|---|---|---|---|
| D0 all off | 282 | 930 | 845 |
| + 2-lane drains | 295 | 978 | 868 |
| + sub_lut slow path | 311 | 1101 | 927 |
| + tbin_tbl | 313 | 1101 | 985 |
| + AVX2 wave U and V | 311 | 1224 | 1070 |
| + 3-pass AVX2 wave | 311 | 1295 | 1148 |
| + AVX2 iRCT (default) | 309 | 1299 | 1155 |
Not taken: fused wave + iRCT without the P buffer (RLH9_DEC_WAVE_MODE=2) is fastest on valid41 (319-324) but slower on
screens (-6..-8%), web8 (-5%) and Kodak (-5..-8%); SIMD 15-threshold search (SLOW_MODE=2) loses to sub_lut.
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
