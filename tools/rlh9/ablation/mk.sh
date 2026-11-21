#!/bin/bash
# mk.sh <tables.h> <out-bench-bin> [extra g++ flags, e.g. -DRLH9_ZTHR=101]
# Builds the llrice bench (ENGINE 9) against the given tables in a temp copy of the repo root.
R=${LLRICE:-$(cd "$(dirname "$0")/../../.." && pwd)}
d=$(mktemp -d); cp "$R"/*.hpp "$R"/*.h "$R"/bench.cpp "$d"/; cp "$1" "$d"/rlh9_tables.h
g++ -O3 -DNDEBUG -march=native -std=c++17 -DLLRICE_ENGINE=9 -DCODER=3 "${@:3}" "$d"/bench.cpp -o "$2"; rc=$?; rm -rf "$d"; exit $rc
