#!/bin/bash
# Rebuild rlh9_tables.h (R8F4) from the image list in train_list.txt.
# Usage: tools/rlh9/train_tables.sh <photo dir> <gb82 dir> <web dir> <game dir> [out.h]
#   each dir holds <stem>.ppm or <stem>.png; PNG needs python3 + PIL (tools/rlh9/png2ppm.py).
# Same command as the committed tables: -f 0 photo, -f 1 -w 4 gb82, -w 4 web, -w 1 game.
set -e
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
[ $# -ge 4 ] || { sed -n 2,5p "$0"; exit 1; }
out=${5:-$root/rlh9_tables.h}; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
g++ -O3 -march=native -std=c++17 -I"$root" "$here/rlh9_train.cpp" -o "$tmp/rlh9_train"
link() {   # link <group> <dir>: symlink every listed stem of the group into $tmp/<group>
  mkdir -p "$tmp/$1"; local n=0
  while IFS=$'\t' read -r g stem; do
    [ "$g" = "$1" ] || continue
    for e in ppm png; do [ -f "$2/$stem.$e" ] && { ln -s "$(cd "$2" && pwd)/$stem.$e" "$tmp/$1/$stem.$e"; n=$((n+1)); break; }; done
  done < <(grep -v '^#' "$here/train_list.txt")
  echo "$1: $n files" >&2
}
link photo "$1"; link gb82 "$2"; link web "$3"; link game "$4"
PNG2PPM="python3 $here/png2ppm.py" "$tmp/rlh9_train" "$out" \
  -f 0 "$tmp/photo" -f 1 -w 4 "$tmp/gb82" -w 4 "$tmp/web" -w 1 "$tmp/game"
