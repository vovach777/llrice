# R8F2 ablation (ENGINE 9), 2026-11-22

Reproduces what each part of R8F2 gives. Claude's sandbox, Xeon ~3.8 GHz, -r 3.
Test sets: test49 = 49 held-out CLIC photos; 12 screens = clic-palic2 halic-wins-screen-p6; web8 = 8 tall QOI web pages
(amazon, apple, cnn, creativecommons, imdb, microsoft, nytimes, reddit). None of them is in any training set below.

    ./mk.sh <tables.h> <bin> [-DRLH9_ZTHR=..]        # build the bench with these tables
    PHOTOS=.. SCREENS12=.. WEB8=.. R=3 ./run.sh <bin>...

Training (tools/rlh9/rlh9_train.cpp, SKIP=test49.txt, PNG2PPM="python3 ../png2ppm.py"):

    rlh9_train t.h -f 0 <CLIC 585 ppm> -f 1 -w 4 <gb82sc: codec_wiki gmessages graph gui> -w 1 <QOI screenshot_game>

## Why the context adapts and why it is fast

R8F1 context = min(kr >> 2, 15) * 3 + min(previous run, 2), kr = llrice adapt_kr state. The decoder computed adapt_kr
(shift, compare, add, clamp) on the value it had just rebuilt from the mantissa: the next LUT index waited for
LUT -> mantissa -> v -> adapt.
R8F2 (Argon's split micro-FSM): st = sv | sr << 6.
- sv, 48 states = kr * 2/3 of the same adapt_kr, but driven by the half-octave class of the value (16 classes),
  which is a function of the Huffman symbol alone. Calibrated: one step of adapt_kr on a representative value per class.
- sr, 16 states = EMA 1/4 of the run class min(bw(run), 7), also a function of the symbol: it sees how long the runs are
  recently, where R8F1 saw only the last run (0, 1, 2+).
- Both classes are stored as one byte next to each LUT entry, so the next state is two table loads
  (v[sv][cls & 15], r[sr][cls >> 4]) that run in parallel with the mantissa work. Decode 173 -> 217 MB/s on photos.

## What each part gives

| R8F1 -> | test49 photos | 12 screens | web8 |
|---|---|---|---|
| R8F1 (kr x run ctx, photo tables) | 8.9807 | 1.4647 | 2.0121 |
| + micro-FSM, photo tables only (RLH9_ZTHR=101) | 8.9913 (+0.12%) | 1.4033 (-4.2%) | 2.0137 (+0.1%) |
| + screen table set per band (R8F2, ZTHR 65) | 8.9882 (+0.08%) | 1.3154 (-10.2%) | 1.9738 (-1.9%) |

The screen tables give more than the FSM: on web8 the FSM alone gives nothing, all of -1.9% is the table set.
On photos the detector sends some flat bands to the screen set, which helps a little (8.9913 -> 8.9882).

## ctx split (48 contexts per group)

Early runs: one split for both sets; tables photos -f 0, screens -f 1 on QOI web6 + QOI games.

| ctx | photos | 12 screens* | web8 |
|---|---|---|---|
| V0 (sv >> 2) * 4 + (sr >> 2), Argon | 9.0050 | 1.2954* | 1.9710 |
| V1 (sv >> 1) * 2 + (sr >> 3) | 8.9905 | 1.3027* | 1.9729 |
| V2 (sv >> 2) * 4 + sr bucket {0-1, 2-4, 5-8, 9-15} | 8.9945 | 1.2994* | 1.9673 |

R8F2 uses V1 for the photo set and V0 for the screen set: (st & m) + ((st >> 6) >> c), (m, c) = (62, 3) / (60, 2).
\* these early screen tables included 6 QOI web pages that are in the 12 screens (leak); compare the columns
inside this table only.

## Other knobs

- Detector threshold (percent of zero residuals in a band): 50 / 58 / 65 / 72 ->
  photos 8.9967 / 8.9915 / 8.9883 / 8.9888, 12 screens 1.2948* / 1.2949* / 1.2955* / 1.2964*, web8 1.9700 / 1.9701 / 1.9711 / 1.9745 (R8F2 split, early tables).
- Training routing (shared V0 split): every band by the detector 9.0137 / 1.2914* / 1.9741; photos forced photo set,
  screens by detector 9.0150 / 1.2898* / 1.9789; both forced 9.0050 / 1.2954* / 1.9710 (chosen).
- Weight 4 for the 4 GUI screens vs 1: 12 screens 1.3161 -> 1.3154, web8 1.9744 -> 1.9738.

## Leaks found

- Argon's data_screens_train are crops of the web8 pages: his web8 numbers are not held out.
- The 12 screens are 6 gb82sc + 6 QOI web pages (duckduckgo, en.wikipedia, news.ycombinator, phoboslab, stripe, sublime);
  training screens on the QOI web set leaks into them (1.2914 dirty vs 1.3154 clean).
