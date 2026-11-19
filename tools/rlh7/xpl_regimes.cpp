// xpl_regimes.cpp - Real static-table Huffman encoder/decoder + bit-exact round-trip verification
// Compares SEP, ONE, GRB, XPL1, XPL2, XPL4 (rmag4), XPL6 (sig(g)xrmag4) on screens12 + clic30
// using 2-fold cross-validation (train static tables on fold A, encode/decode/verify fold B, and vice versa).
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
static inline uint32_t val_from_cv_man(int cv, uint32_t man) {
    if (cv <= 1) return uint32_t(cv);
    if (cv <= 3) return uint32_t(cv); // not used when b>=2 -> cv=4+... wait: bw(2)=2->4, bw(3)=2->5
    if (cv <= 5) return uint32_t(2 + (cv - 4)); // b=2: 2 or 3, 0 mantissa bits
    int b = (cv + 4) / 4;
    int q = (cv + 4) % 4;
    return (1u << (b - 1)) | (uint32_t(q) << (b - 3)) | man;
}
static inline int vm_of_cv(int cv) {
    if (cv <= 5) return 0;
    int b = (cv + 4) / 4;
    return b - 3;
}
static inline int rsym_of(uint32_t rl) { return rl < 8 ? int(rl) : 8 + bw(rl) - 4; }
static inline int rman_bits(int rs) { return rs < 8 ? 0 : (rs - 8 + 3); }
static inline uint32_t rl_from_rs_man(int rs, uint32_t man) {
    if (rs < 8) return uint32_t(rs);
    int b = rs - 8 + 4;
    return (1u << (b - 1)) | man;
}

static inline uint32_t ctx34(uint32_t cr, uint32_t vk) {
    if (!vk) return 0;
    return (std::min(vk, 11u) - 1u) * 3u + (cr == 0u ? 0u : (cr == 1u ? 1u : 2u)) + 1u;
}
constexpr int NV = 44, NR = 19, EOBS = NR * NV, NS = EOBS + 1;
constexpr int MAX_HUFF_LEN = 15;

// Canonical Huffman table (encoder + decoder)
struct HuffTab {
    std::array<uint8_t, NS> len{};
    std::array<uint16_t, NS> code{};
    std::array<uint16_t, 16> first_code{};
    std::array<int, 16> first_sym{};
    std::vector<uint16_t> syms;
    int max_len = 0;

    void build(const std::array<double, NS>& freq) {
        len.fill(0); code.fill(0); syms.clear(); max_len = 0;
        struct Item { double w; int s; };
        std::vector<Item> v;
        for (int i = 0; i < NS; ++i) {
            double w = freq[i] > 0.0 ? freq[i] : 1e-3;
            v.push_back({w, i});
        }
        int m = int(v.size());
        std::vector<int> par(2 * m, -1);
        std::vector<double> w(2 * m);
        using P = std::pair<double, int>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> h;
        for (int i = 0; i < m; ++i) { w[i] = v[i].w; h.push({w[i], i}); }
        int nx = m;
        while (h.size() > 1) {
            auto a = h.top(); h.pop();
            auto b = h.top(); h.pop();
            w[nx] = a.first + b.first;
            par[a.second] = nx; par[b.second] = nx;
            h.push({w[nx], nx}); ++nx;
        }
        std::vector<int> d(m);
        for (int i = 0; i < m; ++i) {
            int k = 0, j = i;
            while (par[j] >= 0) { j = par[j]; ++k; }
            d[i] = std::min(k, MAX_HUFF_LEN);
        }
        auto kraft = [&]() { double k = 0; for (int x : d) k += std::ldexp(1.0, -x); return k; };
        while (kraft() > 1.0 + 1e-12) {
            int bi = -1;
            for (int i = 0; i < m; ++i)
                if (d[i] < MAX_HUFF_LEN && (bi < 0 || v[i].w < v[bi].w)) bi = i;
            d[bi]++;
        }
        int bl[16] = {};
        for (int i = 0; i < m; ++i) {
            len[v[i].s] = uint8_t(d[i]);
            bl[d[i]]++;
            if (d[i] > max_len) max_len = d[i];
        }
        uint32_t c = 0;
        uint32_t nc[16] = {};
        for (int b = 1; b <= MAX_HUFF_LEN; ++b) {
            c = (c + bl[b - 1]) << 1;
            nc[b] = c;
            first_code[b] = uint16_t(c);
        }
        std::vector<std::pair<int, int>> ord;
        ord.reserve(m);
        for (int s = 0; s < NS; ++s) if (len[s]) ord.push_back({len[s], s});
        std::sort(ord.begin(), ord.end());
        int cur_l = 0;
        for (int idx = 0; idx < int(ord.size()); ++idx) {
            int l = ord[idx].first, s = ord[idx].second;
            while (cur_l < l) { ++cur_l; first_sym[cur_l] = idx; }
            code[s] = uint16_t(nc[l]++);
            syms.push_back(uint16_t(s));
        }
        for (int l = cur_l + 1; l <= MAX_HUFF_LEN; ++l) first_sym[l] = int(ord.size());
    }
};

struct BitWriter {
    std::vector<uint8_t> buf;
    uint64_t acc = 0;
    int nb = 0;
    uint64_t total_bits = 0;
    inline void put(uint32_t v, int n) {
        total_bits += n;
        if (n == 0) return;
        acc = (acc << n) | (v & ((1ULL << n) - 1ULL));
        nb += n;
        while (nb >= 8) {
            nb -= 8;
            buf.push_back(uint8_t((acc >> nb) & 0xFF));
        }
    }
    void flush() {
        if (nb > 0) {
            buf.push_back(uint8_t((acc << (8 - nb)) & 0xFF));
            nb = 0; acc = 0;
        }
    }
};

struct BitReader {
    const uint8_t* p;
    size_t sz, pos = 0;
    uint64_t acc = 0;
    int nb = 0;
    BitReader(const std::vector<uint8_t>& b) : p(b.data()), sz(b.size()) {}
    inline void fill(int need) {
        while (nb < need && pos < sz) {
            acc = (acc << 8) | p[pos++];
            nb += 8;
        }
    }
    inline uint32_t get(int n) {
        if (n == 0) return 0;
        fill(n);
        nb -= n;
        return uint32_t((acc >> nb) & ((1ULL << n) - 1ULL));
    }
    inline int dec(const HuffTab& t) {
        uint32_t cw = 0;
        for (int l = 1; l <= t.max_len; ++l) {
            cw = (cw << 1) | get(1);
            int idx = t.first_sym[l] + int(cw) - int(t.first_code[l]);
            int end = (l < t.max_len) ? t.first_sym[l + 1] : int(t.syms.size());
            if (idx >= t.first_sym[l] && idx < end) return t.syms[idx];
        }
        return -1;
    }
};

static inline int gamma_cost(uint32_t x) { return 2 * bw(x) - 1; }
static inline int lv3(uint16_t v) { return v == 0 ? 0 : (v <= 2 ? 1 : 2); }
static inline int lv2(uint16_t v) { return v != 0; }
static inline int rmag4(uint16_t v) { int b = bw(v); return b == 0 ? 0 : (b == 1 ? 1 : (b <= 3 ? 2 : 3)); }

// Mode IDs:
// 0: SEP C512
// 1: ONE ph3 C512
// 2: GRB ph2 C512
// 3: XPL1 sig(g)/(g,r) C512
// 4: XPL2 lv3(g)/(g,r) C512
// 5: XPL4 sig(g)/rmag4(r) C512
// 6: XPL6 sig(g)/sig(g)xrmag4(r) C512
static int xclass(int mode, int plane, uint16_t g, uint16_t r, int& ncls) {
    if (plane == 0) { ncls = 1; return 0; }
    if (plane == 1) {
        if (mode == 4) { ncls = 3; return lv3(g); }
        ncls = 2; return lv2(g);
    }
    if (mode == 3) { ncls = 4; return lv2(g) * 2 + lv2(r); }
    if (mode == 4) { ncls = 9; return lv3(g) * 3 + lv3(r); }
    if (mode == 5) { ncls = 4; return rmag4(r); }
    ncls = 8; return lv2(g) * 4 + rmag4(r);
}

static int num_ctx(int mode) {
    if (mode == 0) return 34;
    if (mode == 1) return 34 * 3;
    if (mode == 2) return 34 * 3; // ch0: 34, rb: 34*2
    if (mode == 3) return 34 * 7; // 1 + 2 + 4
    if (mode == 4) return 34 * 13; // 1 + 3 + 9
    if (mode == 5) return 34 * 7; // 1 + 2 + 4
    return 34 * 11; // 1 + 2 + 8
}

struct ImgData {
    std::string name;
    bool is_screen;
    size_t W, H;
    std::vector<uint16_t> val[3];
};

// If tabs == nullptr, only accumulate histograms; if tabs != nullptr, encode + decode + verify!
static void process_image(int mode, const ImgData& img, std::vector<std::array<double, NS>>* hist,
                          const std::vector<HuffTab>* tabs, double& out_bits, double& out_strp_bits,
                          uint64_t& mismatches) {
    const size_t W = img.W, H = img.H, n = W * H;
    const size_t C = 512;
    BitWriter bw_enc;
    std::vector<double> row_bits(H, 0.0);

    auto emit = [&](uint32_t cx, uint32_t sym, uint32_t man, int nman) {
        if (hist) (*hist)[cx][sym] += 1.0;
        if (tabs) {
            const auto& t = (*tabs)[cx];
            bw_enc.put(t.code[sym], t.len[sym]);
            bw_enc.put(man, nman);
        }
    };

    if (mode == 0) { // SEP C512
        for (int p = 0; p < 3; ++p) {
            for (size_t y = 0; y < H; ++y) {
                uint64_t rb0 = bw_enc.total_bits;
                for (size_t b = 0; b < W; b += C) {
                    size_t e = std::min(W, b + C);
                    uint32_t pcr = 0, pvk = 0, rl = 0;
                    for (size_t x = b; x < e; ++x) {
                        uint32_t v = img.val[p][y * W + x];
                        if (!v) { ++rl; continue; }
                        uint32_t cx = ctx34(pcr, pvk);
                        int rs = rsym_of(rl), cv = cv_of(v);
                        uint32_t sym = uint32_t(rs * NV + cv);
                        int rm = rman_bits(rs), vm = vm_of_cv(cv);
                        uint32_t rman_v = rm ? (rl & ((1u << rm) - 1u)) : 0u;
                        uint32_t vman_v = vm ? (v & ((1u << vm) - 1u)) : 0u;
                        emit(cx, sym, (rman_v << vm) | vman_v, rm + vm);
                        pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v)); rl = 0;
                    }
                    if (rl) emit(ctx34(pcr, pvk), uint32_t(EOBS), 0, 0);
                }
                row_bits[y] += double(bw_enc.total_bits - rb0);
            }
        }
    } else if (mode == 1) { // ONE ph3 C512
        for (size_t y = 0; y < H; ++y) {
            uint64_t rb0 = bw_enc.total_bits;
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                uint32_t pcr = 0, pvk = 0, rl = 0, st = 0;
                for (size_t i = 0; i < (e - b) * 3; ++i) {
                    size_t x = b + i / 3; int p = int(i % 3);
                    uint32_t v = img.val[p][y * W + x];
                    if (!v) { ++rl; continue; }
                    uint32_t cx = ctx34(pcr, pvk) * 3u + (st % 3u);
                    int rs = rsym_of(rl), cv = cv_of(v);
                    uint32_t sym = uint32_t(rs * NV + cv);
                    int rm = rman_bits(rs), vm = vm_of_cv(cv);
                    uint32_t rman_v = rm ? (rl & ((1u << rm) - 1u)) : 0u;
                    uint32_t vman_v = vm ? (v & ((1u << vm) - 1u)) : 0u;
                    emit(cx, sym, (rman_v << vm) | vman_v, rm + vm);
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v)); rl = 0; st = uint32_t(i + 1);
                }
                if (rl) emit(ctx34(pcr, pvk) * 3u + (st % 3u), uint32_t(EOBS), 0, 0);
            }
            row_bits[y] += double(bw_enc.total_bits - rb0);
        }
    } else if (mode == 2) { // GRB ph2 C512
        for (size_t y = 0; y < H; ++y) {
            uint64_t rb0 = bw_enc.total_bits;
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                uint32_t pcr = 0, pvk = 0, rl = 0;
                for (size_t x = b; x < e; ++x) {
                    uint32_t v = img.val[0][y * W + x];
                    if (!v) { ++rl; continue; }
                    uint32_t cx = ctx34(pcr, pvk);
                    int rs = rsym_of(rl), cv = cv_of(v);
                    uint32_t sym = uint32_t(rs * NV + cv);
                    int rm = rman_bits(rs), vm = vm_of_cv(cv);
                    uint32_t rman_v = rm ? (rl & ((1u << rm) - 1u)) : 0u;
                    uint32_t vman_v = vm ? (v & ((1u << vm) - 1u)) : 0u;
                    emit(cx, sym, (rman_v << vm) | vman_v, rm + vm);
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v)); rl = 0;
                }
                if (rl) emit(ctx34(pcr, pvk), uint32_t(EOBS), 0, 0);
            }
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                uint32_t pcr = 0, pvk = 0, rl = 0, st = 0;
                for (size_t i = 0; i < (e - b) * 2; ++i) {
                    size_t x = b + i / 2; int p = 1 + int(i % 2);
                    uint32_t v = img.val[p][y * W + x];
                    if (!v) { ++rl; continue; }
                    uint32_t cx = 34u + ctx34(pcr, pvk) * 2u + (st % 2u);
                    int rs = rsym_of(rl), cv = cv_of(v);
                    uint32_t sym = uint32_t(rs * NV + cv);
                    int rm = rman_bits(rs), vm = vm_of_cv(cv);
                    uint32_t rman_v = rm ? (rl & ((1u << rm) - 1u)) : 0u;
                    uint32_t vman_v = vm ? (v & ((1u << vm) - 1u)) : 0u;
                    emit(cx, sym, (rman_v << vm) | vman_v, rm + vm);
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v)); rl = 0; st = uint32_t(i + 1);
                }
                if (rl) emit(34u + ctx34(pcr, pvk) * 2u + (st % 2u), uint32_t(EOBS), 0, 0);
            }
            row_bits[y] += double(bw_enc.total_bits - rb0);
        }
    } else { // XPL modes 3..6
        int nc1 = 0, nc2 = 0; xclass(mode, 1, 0, 0, nc1); xclass(mode, 2, 0, 0, nc2);
        const int base[3] = {0, 1, 1 + nc1};
        const int ngrp = 1 + nc1 + nc2;
        std::vector<int> ccls(C);
        for (size_t y = 0; y < H; ++y) {
            uint64_t rb0 = bw_enc.total_bits;
            for (int p = 0; p < 3; ++p) {
                for (size_t b = 0; b < W; b += C) {
                    size_t e = std::min(W, b + C);
                    int ncls = 1;
                    for (size_t x = b; x < e; ++x) ccls[x - b] = xclass(mode, p, img.val[0][y * W + x], img.val[1][y * W + x], ncls);
                    for (int sc = 0; sc < ncls; ++sc) {
                        uint32_t pcr = 0, pvk = 0, rl = 0; bool any = false;
                        int grp = base[p] + sc;
                        for (size_t x = b; x < e; ++x) {
                            if (ccls[x - b] != sc) continue;
                            any = true;
                            uint32_t v = img.val[p][y * W + x];
                            if (!v) { ++rl; continue; }
                            uint32_t cx = ctx34(pcr, pvk) * uint32_t(ngrp) + uint32_t(grp);
                            int rs = rsym_of(rl), cv = cv_of(v);
                            uint32_t sym = uint32_t(rs * NV + cv);
                            int rm = rman_bits(rs), vm = vm_of_cv(cv);
                            uint32_t rman_v = rm ? (rl & ((1u << rm) - 1u)) : 0u;
                            uint32_t vman_v = vm ? (v & ((1u << vm) - 1u)) : 0u;
                            emit(cx, sym, (rman_v << vm) | vman_v, rm + vm);
                            pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v)); rl = 0;
                        }
                        if (any && rl) emit(ctx34(pcr, pvk) * uint32_t(ngrp) + uint32_t(grp), uint32_t(EOBS), 0, 0);
                    }
                }
            }
            row_bits[y] += double(bw_enc.total_bits - rb0);
        }
    }

    if (!tabs) return;
    bw_enc.flush();
    out_bits += double(bw_enc.total_bits);
    double strp = 0.0;
    for (size_t y = 0; y < H; ++y) if ((y % 4) != 3) strp += gamma_cost(uint32_t(row_bits[y]) + 1u);
    out_strp_bits += double(bw_enc.total_bits) + strp;

    // ---- Real bit-level decode + verify ----
    BitReader br(bw_enc.buf);
    std::vector<uint16_t> dec[3];
    for (int p = 0; p < 3; ++p) dec[p].assign(n, 0);

    auto dec_tok = [&](uint32_t cx, uint32_t& rl_out, uint16_t& v_out) -> bool {
        int sym = br.dec((*tabs)[cx]);
        if (sym == EOBS) return false;
        int rs = sym / NV, cv = sym % NV;
        int rm = rman_bits(rs), vm = vm_of_cv(cv);
        uint32_t rman_v = br.get(rm), vman_v = br.get(vm);
        rl_out = rl_from_rs_man(rs, rman_v);
        v_out = uint16_t(val_from_cv_man(cv, vman_v));
        return true;
    };

    if (mode == 0) {
        for (int p = 0; p < 3; ++p) {
            for (size_t y = 0; y < H; ++y) {
                for (size_t b = 0; b < W; b += C) {
                    size_t e = std::min(W, b + C);
                    uint32_t pcr = 0, pvk = 0; size_t pos = b;
                    while (pos < e) {
                        uint32_t rl = 0; uint16_t v = 0;
                        if (!dec_tok(ctx34(pcr, pvk), rl, v)) break;
                        pos += rl;
                        dec[p][y * W + pos] = v;
                        pos++;
                        pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v));
                    }
                }
            }
        }
    } else if (mode == 1) {
        for (size_t y = 0; y < H; ++y) {
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                size_t len3 = (e - b) * 3;
                uint32_t pcr = 0, pvk = 0; size_t pos = 0;
                while (pos < len3) {
                    uint32_t rl = 0; uint16_t v = 0;
                    if (!dec_tok(ctx34(pcr, pvk) * 3u + uint32_t(pos % 3), rl, v)) break;
                    pos += rl;
                    dec[int(pos % 3)][y * W + b + pos / 3] = v;
                    pos++;
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v));
                }
            }
        }
    } else if (mode == 2) {
        for (size_t y = 0; y < H; ++y) {
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                uint32_t pcr = 0, pvk = 0; size_t pos = b;
                while (pos < e) {
                    uint32_t rl = 0; uint16_t v = 0;
                    if (!dec_tok(ctx34(pcr, pvk), rl, v)) break;
                    pos += rl;
                    dec[0][y * W + pos] = v;
                    pos++;
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v));
                }
            }
            for (size_t b = 0; b < W; b += C) {
                size_t e = std::min(W, b + C);
                size_t len2 = (e - b) * 2;
                uint32_t pcr = 0, pvk = 0; size_t pos = 0;
                while (pos < len2) {
                    uint32_t rl = 0; uint16_t v = 0;
                    if (!dec_tok(34u + ctx34(pcr, pvk) * 2u + uint32_t(pos % 2), rl, v)) break;
                    pos += rl;
                    dec[1 + int(pos % 2)][y * W + b + pos / 2] = v;
                    pos++;
                    pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v));
                }
            }
        }
    } else {
        int nc1 = 0, nc2 = 0; xclass(mode, 1, 0, 0, nc1); xclass(mode, 2, 0, 0, nc2);
        const int base[3] = {0, 1, 1 + nc1};
        const int ngrp = 1 + nc1 + nc2;
        std::vector<size_t> spos;
        for (size_t y = 0; y < H; ++y) {
            for (int p = 0; p < 3; ++p) {
                for (size_t b = 0; b < W; b += C) {
                    size_t e = std::min(W, b + C);
                    int ncls = 1; xclass(mode, p, 0, 0, ncls);
                    for (int sc = 0; sc < ncls; ++sc) {
                        spos.clear();
                        for (size_t x = b; x < e; ++x) {
                            int nc = 1;
                            if (xclass(mode, p, dec[0][y * W + x], dec[1][y * W + x], nc) == sc) spos.push_back(x);
                        }
                        if (spos.empty()) continue;
                        uint32_t pcr = 0, pvk = 0; size_t idx = 0;
                        int grp = base[p] + sc;
                        while (idx < spos.size()) {
                            uint32_t rl = 0; uint16_t v = 0;
                            if (!dec_tok(ctx34(pcr, pvk) * uint32_t(ngrp) + uint32_t(grp), rl, v)) break;
                            idx += rl;
                            dec[p][y * W + spos[idx]] = v;
                            idx++;
                            pcr = uint32_t(bw(rl)); pvk = uint32_t(bw(v));
                        }
                    }
                }
            }
        }
    }

    for (int p = 0; p < 3; ++p)
        for (size_t i = 0; i < n; ++i)
            if (dec[p][i] != img.val[p][i]) ++mismatches;
}

static std::vector<ImgData> load_dir(const char* dir, bool is_screen) {
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".ppm") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    std::vector<ImgData> res;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        if (img.channel_type != RawImage::ChannelType::uint8) continue;
        SinkPool sink; auto pixels = img.as<uint8_t>();
        for (int c = 0; c < 3; ++c) sink.val[c].reserve(size_t(img.width) * img.height);
        llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
            pixels += 3; return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
        });
        ImgData d; d.name = f; d.is_screen = is_screen; d.W = img.width; d.H = img.height;
        for (int c = 0; c < 3; ++c) d.val[c] = std::move(sink.val[c]);
        res.push_back(std::move(d));
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: xpl_regimes <screens_dir> <clic_dir>\n"); return 1; }
    auto sc = load_dir(argv[1], true);
    auto ph = load_dir(argv[2], false);
    double sc_px = 0, ph_px = 0;
    for (auto& d : sc) sc_px += double(d.W) * d.H;
    for (auto& d : ph) ph_px += double(d.W) * d.H;

    struct Mode { int id; const char* name; };
    const Mode modes[] = {
        {0, "SEP    C512 eob EXACT"},
        {1, "ONE    ph3 C512 eob EXACT"},
        {2, "GRB    ph2 C512 eob EXACT"},
        {3, "XPL1   sig(g)/(g,r) C512"},
        {4, "XPL2   lv3(g)/(g,r) C512"},
        {5, "XPL4   sig(g)/rmag4(r) C512"},
        {6, "XPL6   sig(g)/sig(g)xrmag4"},
    };

    printf("2-fold CV; per fold: S tables = training screens only, P tables = training photos only;\n");
    printf("(a) one combined set [same as step 1], (b) 2 regimes best-of + 1 flag bit, (c) oracle by type:\n");
    printf("%-28s | %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s | %6s\n",
           "config", "a_sc", "a_sc+str", "a_ph", "a_ph+str",
           "b_sc", "b_sc+str", "b_ph", "b_ph+str",
           "c_sc", "c_sc+str", "c_ph", "c_ph+str",
           "ph->S", "sc->P", "mism");
    for (auto& m : modes) {
        int nctx = num_ctx(m.id);
        double a_sc = 0, a_scs = 0, a_ph = 0, a_phs = 0;
        double b_sc = 0, b_scs = 0, b_ph = 0, b_phs = 0;
        double c_sc = 0, c_scs = 0, c_ph = 0, c_phs = 0;
        uint64_t ph_to_S = 0, sc_to_P = 0, mismatches = 0;
        for (int fold = 0; fold < 2; ++fold) {
            double d1 = 0, d2 = 0; uint64_t d3 = 0;

            std::vector<std::array<double, NS>> histA(nctx);
            for (auto& h : histA) h.fill(0.5);
            for (size_t i = 0; i < sc.size(); ++i)
                if (int(i % 2) != fold) process_image(m.id, sc[i], &histA, nullptr, d1, d2, d3);
            for (size_t i = 0; i < ph.size(); ++i)
                if (int(i % 2) != fold) process_image(m.id, ph[i], &histA, nullptr, d1, d2, d3);
            std::vector<HuffTab> tabsA(nctx);
            for (int c = 0; c < nctx; ++c) tabsA[c].build(histA[c]);

            std::vector<std::array<double, NS>> histS(nctx);
            for (auto& h : histS) h.fill(0.5);
            for (size_t i = 0; i < sc.size(); ++i)
                if (int(i % 2) != fold) process_image(m.id, sc[i], &histS, nullptr, d1, d2, d3);
            std::vector<HuffTab> tabsS(nctx);
            for (int c = 0; c < nctx; ++c) tabsS[c].build(histS[c]);

            std::vector<std::array<double, NS>> histP(nctx);
            for (auto& h : histP) h.fill(0.5);
            for (size_t i = 0; i < ph.size(); ++i)
                if (int(i % 2) != fold) process_image(m.id, ph[i], &histP, nullptr, d1, d2, d3);
            std::vector<HuffTab> tabsP(nctx);
            for (int c = 0; c < nctx; ++c) tabsP[c].build(histP[c]);

            for (size_t i = 0; i < sc.size(); ++i) {
                if (int(i % 2) != fold) continue;
                double bA = 0, sA = 0, bS = 0, sS = 0, bP = 0, sP = 0;
                process_image(m.id, sc[i], nullptr, &tabsA, bA, sA, mismatches);
                process_image(m.id, sc[i], nullptr, &tabsS, bS, sS, mismatches);
                process_image(m.id, sc[i], nullptr, &tabsP, bP, sP, mismatches);
                a_sc += bA; a_scs += sA;
                c_sc += bS; c_scs += sS;
                if (bS <= bP) { b_sc += bS + 1; b_scs += sS + 1; }
                else { b_sc += bP + 1; b_scs += sP + 1; ++sc_to_P; }
            }
            for (size_t i = 0; i < ph.size(); ++i) {
                if (int(i % 2) != fold) continue;
                double bA = 0, sA = 0, bS = 0, sS = 0, bP = 0, sP = 0;
                process_image(m.id, ph[i], nullptr, &tabsA, bA, sA, mismatches);
                process_image(m.id, ph[i], nullptr, &tabsS, bS, sS, mismatches);
                process_image(m.id, ph[i], nullptr, &tabsP, bP, sP, mismatches);
                a_ph += bA; a_phs += sA;
                c_ph += bP; c_phs += sP;
                if (bS < bP) { b_ph += bS + 1; b_phs += sS + 1; ++ph_to_S; }
                else { b_ph += bP + 1; b_phs += sP + 1; }
            }
        }
        printf("%-28s | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %8.4f | %8llu %8llu | %6llu\n",
               m.name,
               a_sc / sc_px, a_scs / sc_px, a_ph / ph_px, a_phs / ph_px,
               b_sc / sc_px, b_scs / sc_px, b_ph / ph_px, b_phs / ph_px,
               c_sc / sc_px, c_scs / sc_px, c_ph / ph_px, c_phs / ph_px,
               (unsigned long long)ph_to_S, (unsigned long long)sc_to_P,
               (unsigned long long)mismatches);
        fflush(stdout);
    }
    return 0;
}
