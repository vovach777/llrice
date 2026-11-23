// xpl_model_ext.cpp - extends cross_model.cpp to test r-magnitude chroma contexts for plane b
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <array>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <filesystem>
#include "llrice.hpp"

struct SinkPool { std::vector<uint16_t> val[3]; int opened = 0; };
struct SinkEnc {
    SinkPool* p; int ch;
    explicit SinkEnc(SinkPool& d) : p(&d), ch(d.opened++) {}
    __attribute__((always_inline)) inline void put(uint32_t v) { p->val[ch].push_back(uint16_t(v > 65535u ? 65535u : v)); }
    inline void flush() {}
};
namespace llrice { template <> struct Coders<SinkPool> { using Enc = SinkEnc; }; }

static inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static inline int cv_of(uint32_t v) {
    const int b = bw(v);
    if (b <= 1) return b;
    if (b == 2) return 4 + int(v & 1);
    return 4 * b - 4 + int((v >> (b - 3)) & 3);
}
static inline int vm_of(uint32_t v) { const int b = bw(v); return b >= 3 ? b - 3 : 0; }
static inline uint32_t ctx34(uint32_t cr, uint32_t vk) {
    if (!vk) return 0;
    return (std::min(vk, 11u) - 1u) * 3u + (cr == 0u ? 0u : (cr == 1u ? 1u : 2u)) + 1u;
}
constexpr int NV = 44, NR = 19, EOBS = NR * NV, NS = EOBS + 1, MAXCTX = 34 * 24;

static double huff(const std::array<uint64_t, NS>& f, std::array<uint8_t, NS>& len, int& used) {
    len.fill(0); used = 0;
    std::vector<std::pair<uint64_t,int>> v;
    for (int i = 0; i < NS; ++i) if (f[i]) { v.push_back({f[i], i}); ++used; }
    if (used <= 1) { if (used) { len[v[0].second] = 1; return double(v[0].first); } return 0.0; }
    const int m = used;
    std::vector<int> par(2 * m, -1); std::vector<uint64_t> w(2 * m);
    using P = std::pair<uint64_t,int>; std::priority_queue<P, std::vector<P>, std::greater<P>> h;
    for (int i = 0; i < m; ++i) { w[i] = v[i].first; h.push({w[i], i}); }
    int nx = m;
    while (h.size() > 1) { auto a = h.top(); h.pop(); auto b = h.top(); h.pop(); w[nx] = a.first + b.first; par[a.second] = nx; par[b.second] = nx; h.push({w[nx], nx}); ++nx; }
    std::vector<int> d(m);
    for (int i = 0; i < m; ++i) { int k = 0, j = i; while (par[j] >= 0) { j = par[j]; ++k; } d[i] = std::min(k, 16); }
    auto kraft = [&]() { double k = 0; for (int x : d) k += std::ldexp(1.0, -x); return k; };
    while (kraft() > 1.0 + 1e-12) { int bi = -1; for (int i = 0; i < m; ++i) if (d[i] < 16 && (bi < 0 || v[i].first < v[bi].first)) bi = i; d[bi]++; }
    double bits = 0;
    for (int i = 0; i < m; ++i) { len[v[i].second] = uint8_t(d[i]); bits += double(v[i].first) * d[i]; }
    return bits;
}
static inline int gamma_cost(uint32_t x) { return 2 * bw(x) - 1; }

struct Rec { uint32_t chunk, ctx, sym, man; };
struct Acc { double bits = 0, man = 0, hdr = 0, off = 0, off_stripe = 0, tok = 0; };

static inline int lv3(uint16_t v) { return v == 0 ? 0 : (v <= 2 ? 1 : 2); }
static inline int lv2(uint16_t v) { return v != 0; }
static inline int rmag4(uint16_t v) { int b = bw(v); return b == 0 ? 0 : (b == 1 ? 1 : (b <= 3 ? 2 : 3)); }
static inline int rmag5(uint16_t v) { int b = bw(v); return b <= 3 ? b : 4; } // 0, 1, 2..3, 4..7, >=8

static int xclass(int xm, int plane, uint16_t g, uint16_t r, int& ncls) {
    if (plane == 0) { ncls = 1; return 0; }
    if (plane == 1) {
        switch (xm) {
            case 1: case 3: case 4: case 5: case 6: ncls = 2; return lv2(g);
            case 2: case 7: ncls = 3; return lv3(g);
            default: ncls = 1; return 0;
        }
    }
    switch (xm) {
        case 1: ncls = 4; return lv2(g) * 2 + lv2(r);
        case 2: ncls = 9; return lv3(g) * 3 + lv3(r);
        case 3: ncls = 3; return lv2(g) + lv2(r);
        case 4: ncls = 4; return rmag4(r);                   // b split only by r magnitude (4 classes: 0, 1, 2..7, >=8)
        case 5: ncls = 5; return rmag5(r);                   // b split only by r bw (5 classes: 0, 1, 2, 3, >=4)
        case 6: ncls = 8; return lv2(g) * 4 + rmag4(r);      // b split by sig(g) x rmag4(r) (8 classes)
        case 7: ncls = 3; return lv3(r);                     // b split only by lv3(r) (3 classes: 0, 1..2, >=3)
        default: ncls = 1; return 0;
    }
}

static void run_cross(const std::vector<uint16_t>* v, size_t W, size_t H, size_t C, int xm, bool eob, bool exact, Acc& A) {
    std::vector<Rec> rec; rec.reserve(v[0].size() * 2);
    std::vector<uint32_t> chunk_row, chunk_plane;
    auto rsym = [&](uint32_t rl) { return exact ? (rl < 8 ? rl : 8u + uint32_t(bw(rl)) - 4u) : uint32_t(bw(rl)); };
    auto rman = [&](uint32_t rl) { return exact ? (rl < 8 ? 0u : uint32_t(bw(rl)) - 1u) : (rl > 1 ? uint32_t(bw(rl)) - 1u : 0u); };
    int nc1 = 0, nc2 = 0; xclass(xm, 1, 0, 0, nc1); xclass(xm, 2, 0, 0, nc2);
    const int base[3] = {0, 1, 1 + nc1};
    const size_t cstep = C ? C : W;
    std::vector<uint32_t> ccls(cstep);
    for (size_t y = 0; y < H; ++y) {
        for (int p = 0; p < 3; ++p) {
            for (size_t b = 0; b < W; b += cstep) {
                const size_t e = std::min(W, b + cstep);
                const uint32_t ci = uint32_t(chunk_row.size()); chunk_row.push_back(uint32_t(y)); chunk_plane.push_back(uint32_t(p));
                int ncls = 1;
                for (size_t x = b; x < e; ++x) ccls[x - b] = uint32_t(xclass(xm, p, v[0][y*W+x], v[1][y*W+x], ncls));
                for (int sc = 0; sc < ncls; ++sc) {
                    uint32_t pcr = 0, pvk = 0, rl = 0; bool any = false;
                    const uint32_t grp = uint32_t(base[p] + sc);
                    for (size_t x = b; x < e; ++x) {
                        if (ccls[x - b] != uint32_t(sc)) continue;
                        any = true;
                        const uint32_t val = v[p][y*W+x];
                        if (!val) { ++rl; continue; }
                        const uint32_t cx = ctx34(pcr, pvk) * 24u + grp;
                        const uint32_t sym = std::min(rsym(rl), uint32_t(NR - 1)) * NV + std::min(uint32_t(cv_of(val)), uint32_t(NV - 1));
                        rec.push_back({ ci, cx, sym, rman(rl) + uint32_t(vm_of(val)) });
                        pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(val)); rl = 0;
                    }
                    if (any && rl) {
                        const uint32_t cx = ctx34(pcr, pvk) * 24u + grp;
                        const uint32_t sym = eob ? uint32_t(EOBS) : std::min(rsym(rl), uint32_t(NR - 1)) * NV;
                        rec.push_back({ ci, cx, sym, eob ? 0u : rman(rl) });
                    }
                }
            }
        }
    }
    std::vector<std::array<uint64_t, NS>> hist(MAXCTX);
    for (auto& r : rec) hist[r.ctx][r.sym]++;
    std::vector<std::array<uint8_t, NS>> len(MAXCTX);
    for (int c = 0; c < MAXCTX; ++c) { int used; A.bits += huff(hist[c], len[c], used); if (used) A.hdr += 8 + used * 14.0; }
    std::vector<double> cbits(chunk_row.size(), 0.0);
    for (auto& r : rec) { cbits[r.chunk] += len[r.ctx][r.sym] + r.man; A.man += r.man; A.tok += 1; }
    std::vector<double> rowbits(H, 0.0);
    for (size_t i = 0; i < chunk_row.size(); ++i) {
        rowbits[chunk_row[i]] += cbits[i];
        const bool last = (i + 1 == chunk_row.size()) || chunk_row[i + 1] != chunk_row[i] || chunk_plane[i + 1] != chunk_plane[i];
        if (!last && C) A.off += gamma_cost(uint32_t(cbits[i]) + 1u);
    }
    if (C) for (size_t y = 0; y < H; ++y) if ((y % 4) != 3) A.off_stripe += gamma_cost(uint32_t(rowbits[y]) + 1u);
}

struct Cfg { const char* name; int xm; size_t C; };
static const Cfg CFG[] = {
    { "XPL0 (=SEP)           C512", 0, 512 },
    { "XPL1 sig(g)/(g,r)     C512", 1, 512 },
    { "XPL2 lv3(g)/(g,r)     C512", 2, 512 },
    { "XPL4 rmag4(r)         C512", 4, 512 },
    { "XPL5 rmag5(r)         C512", 5, 512 },
    { "XPL6 sig(g)xrmag4(r)  C512", 6, 512 },
    { "XPL7 lv3(g)/lv3(r)    C512", 7, 512 },
};
constexpr int NC = int(sizeof(CFG) / sizeof(CFG[0]));

int main(int argc, char** argv) {
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(p))
                if (e.path().extension() == ".ppm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end()); files.insert(files.end(), v.begin(), v.end());
        } else files.push_back(p.string());
    }
    double px = 0; std::vector<Acc> A(NC);
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        if (img.channel_type != RawImage::ChannelType::uint8) continue;
        SinkPool sink; auto pixels = img.as<uint8_t>();
        for (int c = 0; c < 3; ++c) sink.val[c].reserve(size_t(img.width) * img.height);
        llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
            pixels += 3; return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
        });
        px += double(img.width) * img.height;
        for (int c = 0; c < NC; ++c) run_cross(sink.val, img.width, img.height, CFG[c].C, CFG[c].xm, true, true, A[c]);
    }
    printf("%-28s %8s %8s %8s %9s %9s\n", "config", "bpp", "payload", "mant", "bpp+ch", "bpp+str");
    for (int c = 0; c < NC; ++c) {
        const Acc& a = A[c];
        printf("%-28s %8.4f %8.4f %8.4f %9.4f %9.4f\n", CFG[c].name,
               (a.bits + a.man + a.hdr) / px, a.bits / px, a.man / px,
               (a.bits + a.man + a.hdr + a.off) / px, (a.bits + a.man + a.hdr + a.off_stripe) / px);
    }
    return 0;
}
