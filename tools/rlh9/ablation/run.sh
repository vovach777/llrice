#!/bin/bash
# run.sh <bench-bin>... ; env PHOTOS, SCREENS12, WEB8 = dirs of .ppm; R = repeats (default 1)
for b in "$@"; do echo -n "$b: "
  for set in "$PHOTOS" "$SCREENS12" "$WEB8"; do "$b" -r "${R:-1}" "$set" 2>&1 | tail -1 | awk '{printf "%s bpp dec %s MB/s | ", $4, $9}'; done; echo; done
