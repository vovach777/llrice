# R8F5 band probe study (Claude, 2026-11-23)

Question: is the zero count in the R8F5 front worth it, and would a per-lane screen flag (Y and S separately, "R8F6") pay?
Companion of the PLS probe study (plt2 branch `pls-probe-2`, hlc/research/pls_probe2) and of the lab note
01d_LAB_PLS_PROBE_2026-11-23.md (Google Drive, lab folder).

- `r8f5_lane_oracle.patch` (on 0448cfb): the encoder codes every band with both table sets and prints
  `BAND PS z zY zS Ylanes(photo) Slanes(photo) Ylanes(screen) Slanes(screen)` to stderr; `lane_oracle.py <dir of logs>`
  compares rules with the per-lane oracle.
- `r8f5_count_cost.patch`: `-DRLH9_FORCE_SC=0|1` fixes the set (same coding path), `-DRLH9_NOCOUNT` removes the
  zero count from the front; the speed difference of the two builds is the cost of counting.

Loss vs the per-lane oracle, % (12 screens / iPhone, private / web8 / 60 DIBCO scans / valid41 / Kodak):
| rule | full12 | iphone | web8 | scans | valid41 | kodak |
|---|---|---|---|---|---|---|
| always photo set | 7.79 | 4.29 | 2.45 | 1.14 | 0.02 | 0.07 |
| always screen set | 0.04 | 0.08 | 0.50 | 1.11 | 1.88 | 1.16 |
| R8F5 (zeros of Y+U+V >= 65%) | 0.03 | 0.45 | 0.23 | 0.76 | 0.06 | 0.07 |
| per-band oracle (one flag) | 0.01 | 0.03 | 0.11 | 0.35 | 0.01 | 0.07 |
| Y+U+V >= 50% (test sets, not a choice) | 0.04 | 0.07 | 0.32 | 0.47 | 0.19 | 0.08 |

- Counting is justified: the flag is worth 1-8%, the count costs ~0-3% of encoding (sandbox, noisy), there is no
  separate cheaper probe in R8F5, and the same count picks the encoder's zero-mask path (+24..48% on screens).
- R8F6 rejected: the two sets share one coder, a wrong set costs 1-7% of a lane (PLS vs PLR: 4-43%); the per-lane
  oracle beats the per-band oracle by 0.35% on scans (PLS: 1.31%) and by ~0 elsewhere.
- Open: threshold 65% looks high (iPhone 0.45% vs 0.08% always-screen); choose it on held-out data, encoder-only.
