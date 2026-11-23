// block_stats.cpp - statistics for LEARNING the regime selector (instead of Argon's hand-tuned
// thresholds).  The regime of block i is a function of counters over block i-1 only -> causal.
// The selector runs once per 32 tokens, so a learned tree there costs ~0 per token.
//
//   block_stats <gridid> <out.bin> <dir:sc|ph> ...
//
// hist[fold][grp][cell][ctx34][symc] uint32, symc = cr*12 + min(vk,11)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

static inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static constexpr int NR = 19, NVK = 12, NSC = NR * NVK, NCTX = 34, NFOLD = 2, NGRP = 2;

static inline int next_ctx_of(int cr, int vk) {
    if (vk <= 0) return 0;
    return (std::min(vk, 11) - 1) * 3 + std::min(cr, 2) + 1;
}
static inline int get_reg(uint32_t s, uint32_t h, uint32_t c) {
    if (s >= 20) { if (h >= 14) return 0; if (h >= 4 || s >= 96) return 1; return 2; }
    if (h >= 16 && s >= 6) return 0;
    if (c < 176) return 3; if (c < 266) return 4; if (c < 356) return 5; if (c < 452) return 6;
    return 7;
}

struct Cnt {
    uint32_t sum_rl = 0, hi_vk = 0, sum_ctx = 0;
    uint32_t sum_vb = 0, max_vb = 0, cnt_rl0 = 0, sum_cr = 0, sum_vk1 = 0;
    inline void add(uint32_t rl, int vb, int cr, int nc) {
        sum_rl += std::min<uint32_t>(rl, 256u);
        hi_vk += (nc >= 16);
        sum_ctx += uint32_t(nc);
        sum_vb += uint32_t(vb);
        max_vb = std::max(max_vb, uint32_t(vb));
        cnt_rl0 += (rl == 0);
        sum_cr += uint32_t(cr);
        sum_vk1 += (vb == 1);
    }
    inline void reset() { *this = Cnt{}; }
};

// exact-threshold quantisers: ENGINE 6's boundaries are representable in these grids
static inline int q_rl(uint32_t s) {
    if (s == 0) return 0; if (s < 6) return 1; if (s < 20) return 2; if (s < 48) return 3;
    if (s < 96) return 4; if (s < 192) return 5; if (s < 512) return 6; return 7;
}
static inline int q_hi(uint32_t h) {
    if (h < 4) return 0; if (h < 8) return 1; if (h < 14) return 2; if (h < 16) return 3;
    if (h < 20) return 4; if (h < 24) return 5; if (h < 28) return 6; return 7;
}
static inline int q_ctx(uint32_t c) {
    if (c < 176) return 0; if (c < 266) return 1; if (c < 356) return 2; if (c < 452) return 3;
    if (c < 600) return 4; if (c < 800) return 5; if (c < 1000) return 6; return 7;
}

// 3 features x 8 levels = 512 cells
static inline int cell_of(int gid, const Cnt& c) {
    switch (gid) {
        case 0: return get_reg(c.sum_rl, c.hi_vk, c.sum_ctx);
        case 1: return (std::min(7, bw(c.sum_rl)) << 6) | (std::min(7, int(c.hi_vk >> 2)) << 3) | std::min(7, int(c.sum_ctx >> 6));
        case 2: return (std::min(7, int(c.sum_vb >> 5)) << 6) | (std::min(7, int(c.cnt_rl0 >> 2)) << 3) | std::min(7, bw(c.sum_rl));
        case 3: return (std::min(7, int(c.sum_vb >> 5)) << 6) | (std::min(7, int(c.max_vb)) << 3) | std::min(7, int(c.cnt_rl0 >> 2));
        case 4: return (std::min(7, bw(c.sum_rl)) << 6) | (std::min(7, int(c.sum_vb >> 5)) << 3) | std::min(7, int(c.sum_cr >> 5));
        case 5: return (std::min(7, int(c.sum_ctx >> 5)) << 6) | (std::min(7, int(c.hi_vk >> 2)) << 3) | std::min(7, int(c.sum_vb >> 5));
        case 6: return (std::min(7, int(c.sum_vb >> 5)) << 6) | (std::min(7, int(c.sum_vk1 >> 2)) << 3) | std::min(7, bw(c.sum_rl));
        case 7: return (std::min(7, bw(c.sum_rl)) << 6) | (std::min(7, int(c.cnt_rl0 >> 2)) << 3) | std::min(7, int(c.sum_ctx >> 6));
        case 8: return (q_rl(c.sum_rl) << 6) | (q_hi(c.hi_vk) << 3) | q_ctx(c.sum_ctx);
        case 9: return (q_rl(c.sum_rl) << 6) | (q_hi(c.hi_vk) << 3) | std::min(7, int(c.sum_vb >> 5));
        case 10: return (q_rl(c.sum_rl) << 6) | (q_ctx(c.sum_ctx) << 3) | std::min(7, int(c.sum_vb >> 5));
        case 11: return (std::min(7, int(c.sum_vb >> 5)) << 6) | (std::min(7, int(c.cnt_rl0 >> 2)) << 3) | q_rl(c.sum_rl);
        case 12: return (q_ctx(c.sum_ctx) << 6) | (q_hi(c.hi_vk) << 3) | std::min(7, int(c.sum_vb >> 5));
        case 13: return (q_rl(c.sum_rl) << 6) | (std::min(7, int(c.sum_vb >> 5)) << 3) | std::min(7, int(c.cnt_rl0 >> 2));
    }
    return 0;
}
static const char* GNAME[] = {
    "engine6_regime", "bw(sum_rl),hi_vk,sum_ctx", "mean_vb,density,bw(sum_rl)",
    "mean_vb,max_vb,density", "bw(sum_rl),mean_vb,mean_cr", "sum_ctx,hi_vk,mean_vb",
    "mean_vb,vk1_rate,bw(sum_rl)", "bw(sum_rl),density,sum_ctx",
    "EXACT rl,hi,ctx", "EXACT rl,hi,mean_vb", "EXACT rl,ctx,mean_vb", "mean_vb,density,EXACT rl",
    "EXACT ctx,hi,mean_vb", "EXACT rl,mean_vb,density"};
static constexpr int NGRID = 14;

struct Tok { uint32_t rl; uint16_t v; uint16_t pad; };

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: block_stats gridid out.bin dir:sc|ph ...\n"); return 1; }
    const int gid = atoi(argv[1]);
    if (gid < 0 || gid >= NGRID) return 1;
    const int ncell = gid == 0 ? 8 : 512;
    const size_t HS = size_t(NFOLD) * NGRP * ncell * NCTX * NSC;
    std::vector<uint32_t> hist(HS, 0);
    fprintf(stderr, "grid %d (%s): %d cells, %.1f MB\n", gid, GNAME[gid], ncell, double(HS * 4) / 1e6);
    size_t tot = 0, used = 0;
    for (int a = 3; a < argc; ++a) {
        std::string spec = argv[a];
        const size_t colon = spec.rfind(':');
        std::string dir = spec.substr(0, colon), kind = spec.substr(colon + 1);
        const bool is_sc = (kind == "sc");
        std::vector<std::string> files;
        for (auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".rlv") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        const double sc_target = 300000.0 * 12.0 / double(files.size() ? files.size() : 1);
        for (auto& f : files) {
            FILE* g = fopen(f.c_str(), "rb");
            if (!g) continue;
            uint32_t hdr[4];
            if (fread(hdr, 4, 4, g) != 4 || hdr[0] != 0x31564c52u) { fclose(g); continue; }
            for (int ch = 0; ch < 3; ++ch) {
                uint64_t n = 0;
                if (fread(&n, 8, 1, g) != 1) break;
                std::vector<Tok> T(n);
                if (fread(T.data(), sizeof(Tok), n, g) != n) break;
                const int grp = (ch == 0) ? 0 : 1;
                const uint32_t w = is_sc ? uint32_t(std::max(1.0, sc_target / double(n))) : 2u;
                Cnt C;
                int cell = gid == 0 ? 1 : cell_of(gid, C);   // first block: counters are zero
                int pctx = 0;
                uint32_t tok_cnt = 0, blk = 0;
                for (uint64_t i = 0; i < n; ++i) {
                    const uint32_t rl = T[i].rl, v = T[i].v;
                    const int vb = bw(v), cr = bw(rl);
                    const bool esc = (rl >= (1u << 18)) || (v >= 2048u);
                    if (blk > 0 && (blk & 3u) == 0u && !esc && cr < NR) {
                        const int fold = int((blk >> 2) & 1u);
                        const int symc = cr * NVK + std::min(vb, 11);
                        hist[(((size_t(fold) * NGRP + grp) * ncell + cell) * NCTX + pctx) * NSC + symc] += w;
                        ++used;
                    }
                    ++tot;
                    if (i + 1 == n && v == 0) break;
                    const int nc = next_ctx_of(cr, vb);
                    C.add(rl, vb, cr, nc);
                    pctx = nc;
                    if ((++tok_cnt & 31u) == 0) {
                        ++blk;
                        cell = cell_of(gid, C);
                        C.reset();
                    }
                }
            }
            fclose(g);
        }
        fprintf(stderr, "  %s (%s): tokens %zu accounted %zu\n", dir.c_str(), kind.c_str(), tot, used);
    }
    FILE* o = fopen(argv[2], "wb");
    uint32_t h[6] = { 0x374b4c42u, NFOLD, uint32_t(ncell), NCTX, NSC, NGRP };
    fwrite(h, 4, 6, o);
    fwrite(hist.data(), 4, HS, o);
    fclose(o);
    printf("grid %d cells %d -> %s (%zu accounted)\n", gid, ncell, argv[2], used);
    return 0;
}
