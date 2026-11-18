// chunk_model.cpp - Claude's task 2026-11-17: row mode with chunks, EOB and offsets.
// Model only: symbol = (run class, quarter-octave value class), raw mantissas, per-image semi-static
// Huffman per context + a header estimate. Values come straight from the llrice front end (CODER=3).
// Prices the chunk cut on the RLH7 context (ctx34/ctx14 x phase), on screens AND photos, with the
// parallel-decoder offsets accounted separately. Also tests Claude's item 4: exact short-run symbols.
//
//   chunk_model <ppm dir|file> ...
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
static const uint8_t C14TAB[34] = {0,2,3,3,4,5,5,6,7,7,8,9,9,8,9,9,10,11,11,10,11,11,10,11,11,12,13,13,12,13,13,12,13,13};
constexpr int NV = 44, NR = 19, EOBS = NR * NV, NS = EOBS + 1, MAXCTX = 34 * 24;

// Huffman with a 16-bit limit; fills len[sym], returns payload bits, used = distinct symbols
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

// One stream. row_vals values per row; each row is cut into chunks of chunk_vals (0 = one chunk total).
static void run_stream(const std::vector<uint16_t>& s, size_t row_vals, size_t chunk_vals, int P,
                       int ctxmode, bool eob, bool exact, bool reset, Acc& A, bool planar = false) {
    std::vector<Rec> rec; rec.reserve(s.size() / 2 + 16);
    std::vector<uint32_t> chunk_row;                 // chunk -> row, to know which is the last in a row
    auto rsym = [&](uint32_t rl) { return exact ? (rl < 8 ? rl : 8u + uint32_t(bw(rl)) - 4u) : uint32_t(bw(rl)); };
    auto rman = [&](uint32_t rl) { return exact ? (rl < 8 ? 0u : uint32_t(bw(rl)) - 1u) : (rl > 1 ? uint32_t(bw(rl)) - 1u : 0u); };
    uint32_t pcr = 0, pvk = 0;
    const size_t n = s.size();
    const size_t rstep = row_vals ? row_vals : n;
    for (size_t rb = 0; rb < n; rb += rstep) {
        const size_t re = std::min(n, rb + rstep);
        const size_t cstep = chunk_vals ? chunk_vals : (re - rb);
        const uint32_t row = uint32_t(rb / rstep);
        for (size_t b = rb; b < re; b += cstep) {
            const size_t e = std::min(re, b + cstep);
            const uint32_t ci = uint32_t(chunk_row.size()); chunk_row.push_back(row);
            if (reset) { pcr = 0; pvk = 0; }
            uint32_t rl = 0; size_t st = b;
            const uint32_t fixed_ph = uint32_t((rb / rstep) % 3);   // planar: the whole chunk is one plane
            for (size_t i = b; i < e; ++i) {
                const uint32_t x = s[i];
                if (!x) { ++rl; continue; }
                const uint32_t vk = uint32_t(bw(x));
                uint32_t cx = ctx34(pcr, pvk); if (ctxmode == 14) cx = C14TAB[cx];
                cx = cx * uint32_t(P) + (planar ? fixed_ph : uint32_t(st % size_t(P)));
                const uint32_t sym = std::min(rsym(rl), uint32_t(NR - 1)) * NV + std::min(uint32_t(cv_of(x)), uint32_t(NV - 1));
                rec.push_back({ ci, cx, sym, rman(rl) + uint32_t(vm_of(x)) });
                pcr = uint32_t(bw(rl)); pvk = vk; rl = 0; st = i + 1;
            }
            if (rl) {
                uint32_t cx = ctx34(pcr, pvk); if (ctxmode == 14) cx = C14TAB[cx];
                cx = cx * uint32_t(P) + (planar ? fixed_ph : uint32_t(st % size_t(P)));
                const uint32_t sym = eob ? uint32_t(EOBS) : std::min(rsym(rl), uint32_t(NR - 1)) * NV;
                rec.push_back({ ci, cx, sym, eob ? 0u : rman(rl) });
                if (!eob) { pcr = uint32_t(bw(rl)); pvk = 0; }
            }
        }
    }
    std::vector<std::array<uint64_t, NS>> hist(MAXCTX);
    for (auto& r : rec) hist[r.ctx][r.sym]++;
    std::vector<std::array<uint8_t, NS>> len(MAXCTX);
    for (int c = 0; c < MAXCTX; ++c) {
        int used; A.bits += huff(hist[c], len[c], used);
        if (used) A.hdr += 8 + used * 14.0;
    }
    std::vector<double> cbits(chunk_row.size(), 0.0);
    for (auto& r : rec) { cbits[r.chunk] += len[r.ctx][r.sym] + r.man; A.man += r.man; A.tok += 1; }
    for (size_t i = 0; i < chunk_row.size(); ++i) {
        const bool last = (i + 1 == chunk_row.size()) || chunk_row[i + 1] != chunk_row[i];
        if (!last && chunk_vals) A.off += gamma_cost(uint32_t(cbits[i]) + 1u);
    }
    // addendum item 5: offsets per STRIPE of 4 rows split into 4 lanes (3 offsets per stripe),
    // instead of one per chunk. Each lane is a whole row, cut at chunk boundaries.
    if (chunk_vals) {
        std::vector<double> rbits;
        for (size_t i = 0; i < chunk_row.size(); ++i) {
            if (i == 0 || chunk_row[i] != chunk_row[i-1]) rbits.push_back(0.0);
            rbits.back() += cbits[i];
        }
        for (size_t i = 0; i < rbits.size(); ++i)
            if ((i % 4) != 3) A.off_stripe += gamma_cost(uint32_t(rbits[i]) + 1u);
    }
}


// ---- Claude's cross-plane model: planes g, r, b of a row, r and b split into sub-streams by the significance of the
// planes already decoded (same row, same position). Runs are counted inside a sub-stream; each sub-stream has its own
// tables x ctx34. One bit stream per (row, plane, chunk): the decoder reads a token of sub-stream s when it reaches a
// position of class s with no pending token. XM selects the class function.
static inline int lv3(uint16_t v) { return v == 0 ? 0 : (v <= 2 ? 1 : 2); }
static inline int lv2(uint16_t v) { return v != 0; }
// returns class id inside the plane and the number of classes
static int xclass(int xm, int plane, uint16_t g, uint16_t r, int& ncls) {
    if (plane == 0) { ncls = 1; return 0; }
    if (plane == 1) {
        switch (xm) { case 1: case 3: ncls = 2; return lv2(g); case 2: ncls = 3; return lv3(g); default: ncls = 1; return 0; }
    }
    switch (xm) {
        case 1: ncls = 4; return lv2(g) * 2 + lv2(r);
        case 2: ncls = 9; return lv3(g) * 3 + lv3(r);
        case 3: ncls = 3; return lv2(g) + lv2(r);          // number of significant planes already known
        default: ncls = 1; return 0;
    }
}
static void run_cross(const std::vector<uint16_t>* v, size_t W, size_t H, size_t C, int xm, bool eob, bool exact, Acc& A) {
    std::vector<Rec> rec; rec.reserve(v[0].size() * 2);
    std::vector<uint32_t> chunk_row, chunk_plane;
    auto rsym = [&](uint32_t rl) { return exact ? (rl < 8 ? rl : 8u + uint32_t(bw(rl)) - 4u) : uint32_t(bw(rl)); };
    auto rman = [&](uint32_t rl) { return exact ? (rl < 8 ? 0u : uint32_t(bw(rl)) - 1u) : (rl > 1 ? uint32_t(bw(rl)) - 1u : 0u); };
    int nc1 = 0, nc2 = 0; xclass(xm, 1, 0, 0, nc1); xclass(xm, 2, 0, 0, nc2);
    const int base[3] = {0, 1, 1 + nc1};            // context group id = base[plane] + class
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

struct Cfg { const char* name; int layout; int C; int ctxmode; bool eob, exact, reset; };
static const Cfg CFG[] = {
    { "SEP    C512 eob EXACT",      0, 512, 34, true,  true,  true  },
    { "ONE    ph3 C512 eob EXACT",  1, 512, 34, true,  true,  true  },
    { "GRB    ph2 C512 eob EXACT",  2, 512, 34, true,  true,  true  },
    { "XPL1 sig(g)/(g,r)    C512",  4, 512, 34, true,  true,  true  },
    { "XPL2 lv3(g)/(g,r)    C512",  5, 512, 34, true,  true,  true  },
    { "XPL3 sig(g)/#sig     C512",  6, 512, 34, true,  true,  true  },
    { "XPL0 =SEP check      C512",  7, 512, 34, true,  true,  true  },
    { "XPL1 sig(g)/(g,r)    C256",  8, 256, 34, true,  true,  true  },
    { "XPL1 sig(g)/(g,r)    C64 ",  9,  64, 34, true,  true,  true  },
    { "XPL1 sig(g)/(g,r)    no cut",10,  0, 34, true,  true,  true  },
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
    std::vector<uint16_t> scratch;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        if (img.channel_type != RawImage::ChannelType::uint8) { fprintf(stderr, "skip %s\n", f.c_str()); continue; }
        SinkPool sink; auto pixels = img.as<uint8_t>();
        for (int c = 0; c < 3; ++c) sink.val[c].reserve(size_t(img.width) * img.height);
        llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
            pixels += 3; return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
        });
        const size_t W = img.width, H = img.height, n = W * H;
        px += double(n);
        scratch.resize(3 * n);
        for (int c = 0; c < NC; ++c) {
            const Cfg& C = CFG[c];
            if (C.layout >= 4) { run_cross(sink.val, W, H, size_t(C.C), C.layout == 4 || C.layout >= 8 ? 1 : C.layout == 5 ? 2 : C.layout == 6 ? 3 : 0, C.eob, C.exact, A[c]); continue; }
            if (C.layout == 0) {
                for (int ch = 0; ch < 3; ++ch) run_stream(sink.val[ch], W, size_t(C.C), 1, C.ctxmode, C.eob, C.exact, C.reset, A[c]);
            } else if (C.layout == 3) {
                for (size_t h = 0; h < H; ++h)
                    for (int c = 0; c < 3; ++c)
                        std::copy_n(sink.val[c].begin() + h*W, W, scratch.begin() + h*3*W + size_t(c)*W);
                run_stream(scratch, W, size_t(C.C), 3, C.ctxmode, C.eob, C.exact, C.reset, A[c], true);
            } else if (C.layout == 1) {
                for (size_t i = 0; i < n; ++i) { scratch[3*i] = sink.val[0][i]; scratch[3*i+1] = sink.val[1][i]; scratch[3*i+2] = sink.val[2][i]; }
                run_stream(scratch, 3 * W, size_t(C.C) * 3, 3, C.ctxmode, C.eob, C.exact, C.reset, A[c]);
            } else {
                for (size_t i = 0; i < n; ++i) { scratch[2*i] = sink.val[1][i]; scratch[2*i+1] = sink.val[2][i]; }
                run_stream(sink.val[0], W, size_t(C.C),     1, C.ctxmode, C.eob, C.exact, C.reset, A[c]);
                scratch.resize(2 * n);
                run_stream(scratch, 2 * W, size_t(C.C) * 2, 2, C.ctxmode, C.eob, C.exact, C.reset, A[c]);
                scratch.resize(3 * n);
            }
        }
    }
    printf("%-28s %8s %8s %8s %9s %9s %9s %9s\n", "config", "bpp", "payload", "mant", "off/chunk", "off/strp", "bpp+ch", "bpp+str");
    for (int c = 0; c < NC; ++c) {
        const Acc& a = A[c];
        printf("%-28s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %9.4f\n", CFG[c].name,
               (a.bits + a.man + a.hdr) / px, a.bits / px, a.man / px, a.off / px, a.off_stripe / px,
               (a.bits + a.man + a.hdr + a.off) / px, (a.bits + a.man + a.hdr + a.off_stripe) / px);
    }
    printf("files %zu  px %.0f\n", files.size(), px);
    return 0;
}
