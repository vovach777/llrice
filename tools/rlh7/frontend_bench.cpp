// frontend_bench.cpp - how much of the encode time is the front-end (RCT + MED + zigzag) alone?
// A sink "encoder" consumes put(v) with one add, so no bit-stream work happens; the loop over the
// image is the same code the real encoder runs.  Compare with llrice-bench enc on the same files.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "llrice.hpp"

struct NullPool { int opened = 0; uint64_t sum = 0; uint64_t nz = 0; };
// LEVEL 0: consume the value with one add (front-end only).
// LEVEL 1: also do the run-length tokenisation and the ENGINE 6/7 context+regime bookkeeping,
//          but no table lookup and no bit writing.
#ifndef FE_LEVEL
#define FE_LEVEL 0
#endif
struct NullEnc {
    NullPool* p;
    uint32_t rl = 0, ctx = 0, reg = 1, sum_rl = 0, hi_vk = 0, sum_ctx = 0, tok_cnt = 0;
    explicit NullEnc(NullPool& d) : p(&d) { ++d.opened; }
    __attribute__((always_inline)) inline void put(uint32_t v) {
#if FE_LEVEL == 0
        p->sum += v; p->nz += (v != 0);
#else
        if (v == 0) { ++rl; return; }
        const uint32_t cr = rl ? (32u - __builtin_clz(rl)) : 0u;
        const uint32_t vk = 32u - __builtin_clz(v);
        p->sum += cr + vk; ++p->nz;
        const uint32_t nc = (std::min(vk, 11u) - 1u) * 3u + (cr == 0u ? 0u : (cr == 1u ? 1u : 2u)) + 1u;
        sum_rl += std::min(rl, 256u); hi_vk += (nc >= 16u); sum_ctx += nc; ctx = nc;
        if ((++tok_cnt & 31u) == 0) {
            reg = llrice::fast_ns::select_regime(sum_rl, hi_vk, sum_ctx);
            sum_rl = hi_vk = sum_ctx = 0;
        }
        rl = 0;
#endif
    }
    inline void flush() { p->sum += rl; }
};
namespace llrice { template <> struct Coders<NullPool> { using Enc = NullEnc; }; }

int main(int argc, char** argv) {
    int reps = 3; std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-r") && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
        std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(p)) if (e.path().extension() == ".ppm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end()); files.insert(files.end(), v.begin(), v.end());
        } else files.push_back(p.string());
    }
    using clk = std::chrono::steady_clock;
    double tb = 0, raw = 0, px = 0; uint64_t S = 0, NZ = 0;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            NullPool sink;
            auto pixels = img.as<uint8_t>();
            auto t0 = clk::now();
            llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
                pixels += 3; return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
            });
            auto t1 = clk::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
            S += sink.sum; NZ += sink.nz;
        }
        double p = double(img.width) * img.height;
        px += p; raw += p * 3; tb += best;
    }
    printf("FRONTEND ONLY %zu files %.1f MB/s  (%.3f ns/px, checksum %llu nz %llu)\n",
           files.size(), raw / tb / 1e6, tb * 1e9 / px, (unsigned long long)S, (unsigned long long)NZ);
    return 0;
}
