// phase_model.cpp - does Claude's "one interleaved stream + phase context" gain survive on top of
// the RLH7 context (34 / 14) and on photos?  Model only: symbol = (run class, quarter-octave value
// class), raw mantissas, per-image semi-static Huffman per context + a header estimate.  No engine
// change, no dumps: the values come straight from the llrice front end through a sink pool.
//
//   phase_model <ppm dir|file> ...
//
// phase = channel index of the position where the current token's zero run STARTS, i.e.
// (index right after the previous non-zero value) mod P.  The decoder tracks it, so it is causal.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <queue>
#include <map>
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
static inline int cv_of(uint32_t v) {                    // quarter octave, as in rlh7
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
static const uint8_t C14TAB[34] = {0,2,3,3,4,5,5,6,7,7,8,9,9,8,9,9,10,11,11,10,11,11,10,11,11,12,13,13,12,13,13,12,13,13};

constexpr int NR = 19, NV = 44, NS = NR * NV;

// Huffman with a 16-bit length limit; returns payload bits, used = number of distinct symbols
static double huff_bits(const std::vector<uint64_t>& f, int& used) {
    used = 0; std::vector<std::pair<uint64_t,int>> v;
    for (int i = 0; i < (int)f.size(); ++i) if (f[i]) { v.push_back({f[i], i}); ++used; }
    if (used <= 1) return used ? double(v[0].first) : 0.0;
    std::vector<int> par(2 * used, -1); std::vector<uint64_t> w(2 * used);
    using P = std::pair<uint64_t,int>; std::priority_queue<P, std::vector<P>, std::greater<P>> h;
    for (int i = 0; i < used; ++i) { w[i] = v[i].first; h.push({w[i], i}); }
    int nx = used;
    while (h.size() > 1) { auto a = h.top(); h.pop(); auto b = h.top(); h.pop(); w[nx] = a.first + b.first; par[a.second] = nx; par[b.second] = nx; h.push({w[nx], nx}); ++nx; }
    std::vector<int> len(used);
    for (int i = 0; i < used; ++i) { int d = 0, j = i; while (par[j] >= 0) { j = par[j]; ++d; } len[i] = d; }
    for (auto& l : len) if (l > 16) l = 16;
    auto kraft = [&]() { double k = 0; for (int l : len) k += std::ldexp(1.0, -l); return k; };
    while (kraft() > 1.0 + 1e-12) { int bi = -1; for (int i = 0; i < used; ++i) if (len[i] < 16 && (bi < 0 || v[i].first < v[bi].first)) bi = i; len[bi]++; }
    double bits = 0; for (int i = 0; i < used; ++i) bits += double(v[i].first) * len[i];
    return bits;
}

struct Tok { uint32_t cr, cv, ph, vk; uint32_t man; };
// rowlayout: 0 = flat stream (phase = st % P), 1 = row-planar (phase = offset/W),
//            2 = row-interleaved (phase = offset%3).  reset = cut the zero run at each row start.
static void tokenize(const std::vector<uint16_t>& s, int P, std::vector<Tok>& t,
                     int rowlayout = 0, size_t W = 0, bool reset = false,
                     const std::vector<uint8_t>* rowlay = nullptr) {
    t.clear(); uint32_t rl = 0; size_t idx = 0, st = 0; bool start = true;
    const size_t rowlen = 3 * W;
    auto phase_of = [&](size_t pos) -> uint32_t {
        if (!rowlayout) return uint32_t(pos % size_t(P));
        const size_t o = pos % rowlen;
        const int rl_ = rowlay ? (*rowlay)[pos / rowlen] : rowlayout;
        return uint32_t(rl_ == 1 ? o / W : o % 3);
    };
    for (uint32_t x : s) {
        if (rowlayout && reset && W && idx && (idx % rowlen) == 0 && rl) {
            const uint32_t cr = uint32_t(bw(rl));
            t.push_back({ cr, 0, phase_of(st), 0, cr > 1 ? cr - 1 : 0 });
            rl = 0; start = true;
        }
        if (start) { st = idx; start = false; }
        ++idx;
        if (!x) { ++rl; continue; }
        const uint32_t cr = uint32_t(bw(rl)), vk = uint32_t(bw(x));
        t.push_back({ cr, uint32_t(cv_of(x)), phase_of(st), vk,
                      (cr > 1 ? cr - 1 : 0) + uint32_t(vm_of(x)) });
        rl = 0; start = true;
    }
    if (rl) { const uint32_t cr = uint32_t(bw(rl)); t.push_back({ cr, 0, phase_of(st), 0, cr > 1 ? cr - 1 : 0 }); }
}

// exact run-mantissa cost of one row under a layout (the term that differs between layouts)
static double row_man_cost(const std::vector<uint16_t>* v, size_t off, size_t W, bool interl) {
    double bits = 0; uint32_t rl = 0;
    auto feed = [&](uint32_t x) { if (!x) { ++rl; return; } if (rl > 1) bits += bw(rl) - 1; rl = 0; };
    if (interl) { for (size_t i = 0; i < W; ++i) for (int c = 0; c < 3; ++c) feed(v[c][off + i]); }
    else { for (int c = 0; c < 3; ++c) { rl = 0; for (size_t i = 0; i < W; ++i) feed(v[c][off + i]); if (rl > 1) bits += bw(rl) - 1; rl = 0; } }
    if (interl && rl > 1) bits += bw(rl) - 1;
    return bits;
}
struct Cfg { const char* name; int layout; int ctxmode; int phase; int rowlayout; bool reset; };
// layout 0 = g,r,b separate; 1 = g + (r,b); 2 = one stream
static const Cfg CFG[] = {
    { "separate   ctx34 (base)", 0, 34, 0, 0, false },
    { "g+(r,b)    ctx34 x ph2",  1, 34, 1, 0, false },
    { "one interl ctx34 x ph3",  2, 34, 1, 0, false },
    { "ROW planar ctx34 x pl3",  3, 34, 1, 1, true  },
    { "ROW planar ctx34 noreset",3, 34, 1, 1, false },
    { "ROW planar ctx34 no ph",  3, 34, 0, 1, true  },
    { "ROW interl ctx34 x ph3",  3, 34, 1, 2, true  },
    { "ROW interl ctx14 x ph3",  3, 14, 1, 2, true  },
    { "ROW ADAPT  ctx34 x ph3",  4, 34, 1, 1, true  },
    { "ROW ADAPT  ctx14 x ph3",  4, 14, 1, 1, true  },
    { "ROW COST   ctx34 x ph3",  5, 34, 1, 1, true  },
    { "ROW COST   ctx14 x ph3",  5, 14, 1, 1, true  },
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
    double px = 0, bits[NC] = {}, man[NC] = {}, hdr[NC] = {}; double tok[NC] = {};
    std::vector<uint16_t> inter; std::vector<Tok> t; std::vector<uint8_t> rowlay;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        if (img.channel_type != RawImage::ChannelType::uint8) { fprintf(stderr, "skip %s\n", f.c_str()); continue; }
        SinkPool sink; auto pixels = img.as<uint8_t>();
        for (int c = 0; c < 3; ++c) sink.val[c].reserve(size_t(img.width) * img.height);
        llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
            pixels += 3; return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
        });
        const size_t n = sink.val[0].size();
        px += double(n);
        for (int c = 0; c < NC; ++c) {
            const Cfg& C = CFG[c];
            std::vector<const std::vector<uint16_t>*> parts;
            int P = 1;
            if (C.layout == 0) { parts = { &sink.val[0], &sink.val[1], &sink.val[2] }; P = 1; }
            else if (C.layout == 1) {
                inter.resize(2 * n);
                for (size_t i = 0; i < n; ++i) { inter[2*i] = sink.val[1][i]; inter[2*i+1] = sink.val[2][i]; }
                parts = { &sink.val[0], &inter }; P = 1;   // phase only inside the chroma stream
            } else if (C.layout == 2) {
                inter.resize(3 * n);
                for (size_t i = 0; i < n; ++i) { inter[3*i] = sink.val[0][i]; inter[3*i+1] = sink.val[1][i]; inter[3*i+2] = sink.val[2][i]; }
                parts = { &inter }; P = 3;
            } else if (C.layout == 4 || C.layout == 5) {
                // per-row layout, chosen CAUSALLY from the previous row's non-zero fraction:
                // sparse rows -> interleave (merge the zero runs), dense rows -> planar (keep them).
                const size_t W = img.width, H = img.height;
                inter.resize(3 * n); rowlay.assign(H, 1);
                for (size_t h = 0; h < H; ++h) {
                    if (h) {
                        if (C.layout == 5) {
                            // causal and exact: price both layouts on the PREVIOUS row, which the
                            // decoder already has, and use the cheaper one for this row.
                            const double cp = row_man_cost(sink.val, (h-1)*W, W, false);
                            const double ci = row_man_cost(sink.val, (h-1)*W, W, true);
                            rowlay[h] = (ci < cp) ? 2 : 1;
                        } else {
                            size_t nz = 0;
                            for (int c = 0; c < 3; ++c)
                                for (size_t i = 0; i < W; ++i) nz += (sink.val[c][(h-1)*W + i] != 0);
                            rowlay[h] = (double(nz) / double(3 * W) < 0.30) ? 2 : 1;
                        }
                    } else rowlay[h] = 1;                                          // first row: planar
                    if (rowlay[h] == 1)
                        for (int c = 0; c < 3; ++c)
                            std::copy_n(sink.val[c].begin() + h*W, W, inter.begin() + h*3*W + size_t(c)*W);
                    else
                        for (size_t i = 0; i < W; ++i)
                            for (int c = 0; c < 3; ++c) inter[h*3*W + 3*i + size_t(c)] = sink.val[c][h*W + i];
                }
                parts = { &inter }; P = 3;
            } else {
                const size_t W = img.width, H = img.height;
                inter.resize(3 * n);
                if (C.rowlayout == 1) {
                    for (size_t h = 0; h < H; ++h)
                        for (int c = 0; c < 3; ++c)
                            std::copy_n(sink.val[c].begin() + h * W, W, inter.begin() + h * 3 * W + size_t(c) * W);
                } else {
                    for (size_t h = 0; h < H; ++h)
                        for (size_t i = 0; i < W; ++i)
                            for (int c = 0; c < 3; ++c) inter[h * 3 * W + 3 * i + size_t(c)] = sink.val[c][h * W + i];
                }
                parts = { &inter }; P = 3;
            }
            for (size_t si = 0; si < parts.size(); ++si) {
                const int Peff = (C.layout == 1 && si == 0) ? 1 : (C.layout == 1 ? 2 : P);
                tokenize(*parts[si], C.phase ? Peff : 1, t, C.layout >= 3 ? C.rowlayout : 0, img.width, C.reset, C.layout >= 4 ? &rowlay : nullptr);
                tok[c] += double(t.size());
                std::map<uint32_t, std::vector<uint64_t>> fr;
                uint32_t pcr = 0, pvk = 0;
                for (auto& x : t) {
                    man[c] += x.man;
                    uint32_t cx = ctx34(pcr, pvk);
                    if (C.ctxmode == 14) cx = C14TAB[cx];
                    if (C.phase) cx = cx * uint32_t(Peff) + x.ph;
                    auto& v = fr[cx]; if (v.empty()) v.assign(NS, 0);
                    v[std::min(x.cr, uint32_t(NR-1)) * NV + std::min(x.cv, uint32_t(NV-1))]++;
                    pcr = x.cr; pvk = x.vk;
                }
                for (auto& kv : fr) { int used; bits[c] += huff_bits(kv.second, used); hdr[c] += 8 + used * 14.0; }
            }
        }
    }
    printf("%-26s %9s %9s %9s %9s %9s\n", "config", "bpp", "payload", "mantissa", "header", "tok/px");
    for (int c = 0; c < NC; ++c)
        printf("%-26s %9.4f %9.4f %9.4f %9.4f %9.4f\n", CFG[c].name,
               (bits[c] + man[c] + hdr[c]) / px, bits[c] / px, man[c] / px, hdr[c] / px, tok[c] / px);
    printf("files %zu  px %.0f\n", files.size(), px);
    return 0;
}
