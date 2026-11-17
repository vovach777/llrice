# LLR - Simple solution good compression

LLRice is streming frendly intraframe lossless image compressor.

## Key Features

- **Lossless Compression**: Every pixel of the original image is preserved during compression and decompression.
- **Advanced Algorithms**: RGB -> RCT -> LOCO-I predictor -> error -> RLGR -> Interleaved Bitstream
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux.

## Why LLRice?

 Simple solution good compression and speed

## Performance Metrics

| File                                | Original        | LLRice          | BPP    | FFV1            | BPP    | HALIC           | BPP    | Status     |
|-------------------------------------|-----------------|-----------------|--------|-----------------|--------|-----------------|--------|------------|
| caroline-sada-92.pnm                | 2,001,016       | 936,480         | 11.23  | 899,969         | 10.79  | 943,842         | 11.32  | OK         |
| paul-jarvis-217.pnm                 | 3,123,766       | 1,571,248       | 12.07  | 1,506,908       | 11.58  | 1,543,681       | 11.86  | OK         |
| sven-schlager-758.pnm               | 1,843,215       | 1,066,392       | 13.89  | 1,018,148       | 13.26  | 1,056,744       | 13.76  | OK         |
| charles-s-524.pnm                   | 8,386,577       | 1,676,464       | 4.80   | 1,576,682       | 4.51   | 1,726,836       | 4.94   | OK         |
| gabe-rodriguez-110.pnm              | 7,594,001       | 2,620,424       | 8.28   | 2,470,181       | 7.81   | 2,729,027       | 8.62   | OK         |
| desi-mendoza-451.pnm                | 8,362,001       | 1,948,024       | 5.59   | 1,748,543       | 5.02   | 1,926,371       | 5.53   | OK         |
| israel-sundseth-1687.pnm            | 8,374,289       | 3,650,936       | 10.46  | 3,526,666       | 10.11  | 3,630,831       | 10.41  | OK         |
| tim-gouw-73090.pnm                  | 9,001,514       | 2,799,216       | 7.46   | 2,556,677       | 6.82   | 2,886,566       | 7.70   | OK         |
| nick-scheerbart-15636.pnm           | 8,306,705       | 4,192,104       | 12.11  | 4,107,186       | 11.87  | 4,184,756       | 12.09  | OK         |
| ryan-pohanic-13342.pnm              | 8,386,577       | 3,874,664       | 11.09  | 3,765,623       | 10.78  | 3,845,120       | 11.00  | OK         |
| jason-ortego-5386.pnm               | 8,386,577       | 4,693,176       | 13.43  | 4,448,798       | 12.73  | 4,687,924       | 13.42  | OK         |
| ray-hennessy-118048.pnm             | 8,374,289       | 2,838,888       | 8.14   | 2,615,088       | 7.49   | 2,722,148       | 7.80   | OK         |
| wim-peters-274.pnm                  | 2,422,969       | 739,224         | 7.32   | 695,569         | 6.89   | 744,722         | 7.38   | OK         |
| jessica-polar-537.pnm               | 7,558,289       | 2,024,552       | 6.43   | 1,909,881       | 6.06   | 2,016,977       | 6.40   | OK         |
| paul-morris-171434.pnm              | 8,982,545       | 3,494,648       | 9.34   | 3,241,596       | 8.66   | 3,391,754       | 9.06   | OK         |
| chris-adams-237.pnm                 | 3,779,152       | 1,386,776       | 8.81   | 1,318,383       | 8.37   | 1,391,034       | 8.83   | OK         |
| **TOTAL**                           | **104,883,482** | **39,513,216**  | **9.04** | **37,405,898** | **8.56** | **39,428,333** | **9.02** |            |

*Results as of January 2026, comparing LLR with FFV1 and HALIC.*

## Speed (16 files):

| Operation            | Time (s)     | Speed (MB/s)    |
|----------------------|--------------|-----------------|
| LLR Compression      | 2.3537       | 42.50           |
| LLR Decompression    | 2.2845       | 43.78           |

## Entropy engines (`LLRICE_ENGINE`)

The predictor, the colour transform and the container are shared; `-DLLRICE_ENGINE=N` picks the
entropy backend. Each engine writes its own revision nibble, so a stream can only be read by the
engine that wrote it (a mismatch is rejected by the header CRC).

| `N` | backend | notes |
|-----|---------|-------|
| 0 | RLGR | reference Microsoft-style Rice/Golomb adaptation |
| 1 | RLGR fast | same bitstream, cheaper adaptation |
| 2, 3, 5 | RLGR variants | page/lane/byte-chunk bitstream experiments |
| 6 | RLH | (run class, value class) symbols, raw mantissas, static Huffman conditioned on the previous symbol |
| **7** | **RLH7 fast** | RLH with retrained tables: 8 regimes × 14 contexts, trained regime selector |
| **8** | **RLH7 strong** | RLH with retrained tables: 24 regimes × 34 contexts, trained regime selector |

Engines 6/7/8 require `-DCODER=3`. Measured on this machine (gcc 13, `-O3 -DNDEBUG -march=native`,
2 cores, best of repeated interleaved runs; **absolute MB/s are machine-specific**):

**Screens — 12 images from `halic-wins-screen-p6`, 74 967 858 px**

| engine | bits/pixel | vs RLGR | encode MB/s | decode MB/s |
|--------|-----------|---------|-------------|-------------|
| 0 RLGR | 1.6014 | – | 279.0 | 121.2 |
| 1 RLGR fast | 1.6014 | – | 259.3 | 116.9 |
| 6 RLH | 1.3604 | −15.05 % | 304.7 | 134.5 |
| **7 RLH7 fast** | **1.3454** | **−15.99 %** | **308.9** | **135.5** |
| **8 RLH7 strong** | **1.3123** | **−18.05 %** | 298.3 | 130.6 |

**Photos — 117 CLIC test images, 271 140 751 px**

| engine | bits/pixel | vs RLGR | encode MB/s | decode MB/s |
|--------|-----------|---------|-------------|-------------|
| 0 RLGR | 8.8448 | – | 78.5 | 51.3 |
| 1 RLGR fast | 8.8448 | – | 81.3 | 47.6 |
| 6 RLH | 8.8089 | −0.41 % | 127.8 | 64.1 |
| **7 RLH7 fast** | 8.8316 | −0.15 % | 127.1 | 63.7 |
| **8 RLH7 strong** | **8.7942** | **−0.57 %** | 125.3 | 61.0 |

Static table footprint and construction time (`tools/rlh7/init_bench.cpp`, best of 5):

| engine | `sizeof(GlobalTables)` | init |
|--------|------------------------|------|
| 6 | 4 842 928 B | 1.83 ms |
| **7** | **2 003 888 B** | **0.82 ms** |
| 8 | 14 495 664 B | 7.53 ms |

Engine 7 is the drop-in choice: smaller and faster to initialise than 6, better on both corpora.
Engine 8 trades ~3× the table memory and ~4× the init time for the best ratio. See
[`tools/rlh7/README.md`](tools/rlh7/README.md) for how the tables are trained.

## Installation

### Prerequisites

- Compiler: Tested with **Intel ICX 2025** on Windows. mingw. clang.
- CMake (Version 3.15 or higher).
- Git.

### Steps


1. After building, two executables will be generated:
   - `llrice-c(.exe)` – The **compressor** tool.
   - `llrice-d(.exe)` – The **decompressor** tool.

## Usage

For examples of how to use LLRice, please refer to the [examples directory](examples/).

### Tools

- **Compressor**: Use the `llrice-c(.exe)` executable to compress images.
- **Decompressor**: Use the `llrice-d(.exe)` executable to decompress images.

## Technical Details

### Core Algorithms

- **RLGR**: by microsoft....
- **Iterleaved Deterministic BitStream**: auto maneged BitStreams.
- **Prediction**: same FFV1

## Development Environment

- **Compiler**: Intel ICX 2025 (Windows).
- **Build System**: CMake (cross-platform support).
- **Dependencies**: No.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! If you'd like to contribute to LLRice, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed updates.

## Acknowledgments

- Inspired by the **FFV1 codec** from FFmpeg.
- Special thanks to the open-source community for their support and feedback.
