// rlh9.hpp - LLRICE_ENGINE 9: RLH8 superscalar photo codec, format R8F1 (signature rev '9').
// Design: Argon (rlh8f.cpp, 2026-11-21) on top of Claude's RLH8s prototype; consolidated from the RLH7/RLH8 lab work.
//
// Front end: RCT (U = r-g, V = b-g, Y = g + ((U+V)>>2)), MED per plane (row 0: left, column 0: top), zigzag.
// Geometry: 8-row bands; each band row of each plane is one chunk of width W; EOB = "rest of this row is zero".
// Lanes: 3 independent byte streams per band (lane = plane Y, U, V); band header = 3 x u32 lane byte sizes.
// Alphabet (E8): 19 run classes (0..7 exact, 8..18 octave + mantissa) x 44 quarter-octave value classes + EOB = 837.
// Context: implicit, per lane: kr = llrice RLGR value-k state (adapt_kr, 0..240) and cr = min(previous run, 2);
//          ctx = min(kr >> 2, 15) * 3 + cr; 2 table groups (Y, shared U/V) -> 96 static canonical Huffman tables (<= 15 bits).
//          State starts at (16, 0) once per image and is carried across rows and bands.
// Decoder: per band an entropy pass with the 3 lanes interleaved ONE TOKEN per lane per round (packed 32-bit LUT,
//          one load per token, branchless adapt), then a fused pass unzigzag + branchless MED + inverse RCT.
// Limits: 8-bit RGB only, width <= 16384 (run classes end at 16383).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#ifndef RLH9_NO_TABLES
#include "rlh9_tables.h"
#endif

namespace rlh9
{
    inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
    inline int cv_of(uint32_t v) { const int b = bw(v); if (b <= 1) return b; if (b == 2) return 4 + int(v & 1); return 4 * b - 4 + int((v >> (b - 3)) & 3); }
    // value classes 2, 3, 6, 7 never occur (cv_of skips them); they get harmless values so that corrupt streams stay defined
    inline int vm_of_cv(int cv) { return cv <= 7 ? 0 : (cv + 4) / 4 - 3; }
    inline uint32_t vbase_of_cv(int cv) { if (cv <= 1) return uint32_t(cv); if (cv == 4 || cv == 5) return uint32_t(cv - 2); if (cv < 8) return 1; int b = (cv + 4) / 4, q = (cv + 4) % 4; return (1u << (b - 1)) | (uint32_t(q) << (b - 3)); }
    inline int rs_of(uint32_t rl) { return rl < 8 ? int(rl) : 8 + bw(rl) - 4; }
    inline int rm_of_rs(int rs) { return rs < 8 ? 0 : rs - 8 + 3; }
    inline uint32_t rbase_of_rs(int rs) { return rs < 8 ? uint32_t(rs) : (1u << (rs - 8 + 3)); }

    constexpr int NV = 44, NR = 19, NSYM = NR * NV + 1, EOB = NSYM - 1, MAXL = 15;
    constexpr int KR_SHIFT = 2, KR_MAX = 15, NCTX = (KR_MAX + 1) * 3, NGRP = 2, TOTC = NGRP * NCTX;
    constexpr uint32_t MAX_W = 16383;
#ifndef RLH9_LUTB
#define RLH9_LUTB 9
#endif
    constexpr int LUTB = RLH9_LUTB;      // decoder-only knob, the format does not depend on it
#ifndef RLH9_NO_TABLES
    static_assert(RLH9_NSYM == NSYM && RLH9_TOTC == TOTC, "rlh9_tables.h does not match rlh9.hpp");
#endif

    inline uint32_t grp_of(int p) { return p ? 1u : 0u; }
    inline uint32_t ctx_of(uint32_t kr, uint32_t cr) { const uint32_t k = kr >> KR_SHIFT; return (k > uint32_t(KR_MAX) ? uint32_t(KR_MAX) : k) * 3u + cr; }
    inline void adapt(uint32_t& kr, uint32_t v) {          // llrice adapt_kr without data-dependent branches
        const uint32_t x = (v - 1) >> (kr >> 3);
        const uint32_t add = x & (0u - uint32_t(x >= 2)), sub = std::min(kr, 2u) & (0u - uint32_t(x == 0));
        kr = std::min(kr + add, 240u) - sub;
    }
    inline int med(int a, int b, int c) { const int mx = a > b ? a : b, mn = a > b ? b : a; return std::max(mn, std::min(mx, a + b - c)); }
    inline uint16_t zz(int e) { return uint16_t((uint32_t(e) << 1) ^ uint32_t(e >> 31)); }
    inline int uz(uint32_t u) { return int(u >> 1) ^ -int(u & 1); }

    struct SlowSym { uint16_t rbase, vbase, vmask; uint8_t vm, nm, eob, len; };
    struct Tables {
        uint8_t len[TOTC][NSYM];
        uint32_t enc_cw[TOTC][NSYM];                     // code | len << 16
        uint32_t lut[TOTC][1 << LUTB];                   // len:4 nm:5 vm:3 vbase:10 rbase:6 ... eob:bit31; 0 = slow path
        uint16_t first[TOTC][MAXL + 2], cntl[TOTC][MAXL + 2], offs[TOTC][MAXL + 2], sorted[TOTC][NSYM];
        SlowSym si[NSYM];
        struct VEnc { uint8_t cv, vm; uint16_t vman; } venc[1024];
        struct REnc { uint8_t rs, rm; uint16_t rman; } renc[256];

        void build(const uint8_t* packed_len) {          // packed_len: TOTC x ceil(NSYM/2) nibbles, low nibble first
            constexpr int PB = (NSYM + 1) / 2;
            for (int tc = 0; tc < TOTC; ++tc) for (int s = 0; s < NSYM; ++s) { const uint8_t b = packed_len[tc * PB + s / 2]; len[tc][s] = uint8_t(s & 1 ? b >> 4 : b & 15); }
            for (int s = 0; s < NSYM; ++s) {
                if (s == EOB) { si[s] = {0, 0, 0, 0, 0, 1, 0}; continue; }
                const int rs = s / NV, cv = s % NV; const uint8_t vm = uint8_t(vm_of_cv(cv));
                si[s] = {uint16_t(rbase_of_rs(rs)), uint16_t(vbase_of_cv(cv)), uint16_t((1u << vm) - 1u), vm, uint8_t(rm_of_rs(rs) + vm), 0, 0};
            }
            for (int tc = 0; tc < TOTC; ++tc) {
                const uint8_t* L = len[tc]; uint32_t c = 0, ns = 0;
                std::memset(lut[tc], 0, sizeof lut[tc]);
                for (int l = 1; l <= MAXL; ++l) {
                    first[tc][l] = uint16_t(c); offs[tc][l] = uint16_t(ns); cntl[tc][l] = 0;
                    for (int s = 0; s < NSYM; ++s) if (L[s] == l) {
                        enc_cw[tc][s] = c | (uint32_t(l) << 16); sorted[tc][ns++] = uint16_t(s); ++cntl[tc][l];
                        if (l <= LUTB) {
                            uint32_t e = 0;
                            if (s == EOB) e = (1u << 31) | uint32_t(l);
                            else if (si[s].rbase < 64 && si[s].vbase < 1024 && si[s].vm <= 7)
                                e = uint32_t(l) | uint32_t(si[s].nm) << 4 | uint32_t(si[s].vm) << 9 | uint32_t(si[s].vbase) << 12 | uint32_t(si[s].rbase) << 22;
                            if (e) for (uint32_t k = c << (LUTB - l); k < (c + 1) << (LUTB - l); ++k) lut[tc][k] = e;
                        }
                        ++c;
                    }
                    c <<= 1;
                }
                if (ns != uint32_t(NSYM)) throw std::runtime_error("rlh9: incomplete code table");
            }
            for (int v = 1; v < 1024; ++v) { const int cv = cv_of(uint32_t(v)); venc[v] = {uint8_t(cv), uint8_t(vm_of_cv(cv)), uint16_t(uint32_t(v) - vbase_of_cv(cv))}; }
            for (int r = 0; r < 256; ++r) { const int rs = rs_of(uint32_t(r)); renc[r] = {uint8_t(rs), uint8_t(rm_of_rs(rs)), uint16_t(uint32_t(r) - rbase_of_rs(rs))}; }
        }
    };
#ifndef RLH9_NO_TABLES
    inline const Tables& tables() { static const Tables* t = [] { auto* p = new Tables; p->build(&RLH9_LEN[0][0]); return p; }(); return *t; }
#else
    const Tables& tables();
#endif

    // ---------------- front end: one 8-row band of residuals (zigzag), planes Y, U, V ----------------
    struct Front {
        int W; std::vector<int16_t> prev[3], cur[3];
        explicit Front(int w) : W(w) { for (int p = 0; p < 3; ++p) { prev[p].assign(size_t(w), 0); cur[p].assign(size_t(w), 0); } }
        // rgb: row-major 8-bit RGB, rows y0..y0+nb-1 -> u[p * nb * W + r * W + x]
        void band(const uint8_t* rgb, int y0, int nb, uint16_t* u) {
            const size_t PS = size_t(nb) * W;
            for (int r = 0; r < nb; ++r) {
                const int y = y0 + r; const uint8_t* s = rgb + size_t(y) * W * 3;
                int16_t *C0 = cur[0].data(), *C1 = cur[1].data(), *C2 = cur[2].data();
                const int16_t *Q0 = prev[0].data(), *Q1 = prev[1].data(), *Q2 = prev[2].data();
                uint16_t *U0 = u + size_t(r) * W, *U1 = U0 + PS, *U2 = U1 + PS;
                int a0 = 0, a1 = 0, a2 = 0;
                for (int x = 0; x < W; ++x) {
                    const int R = s[3 * x], G = s[3 * x + 1], B = s[3 * x + 2], U = R - G, V = B - G, Y = G + ((U + V) >> 2);
                    C0[x] = int16_t(Y); C1[x] = int16_t(U); C2[x] = int16_t(V);
                    int p0, p1, p2;
                    if (y == 0) { p0 = a0; p1 = a1; p2 = a2; }
                    else if (x == 0) { p0 = Q0[0]; p1 = Q1[0]; p2 = Q2[0]; }
                    else { p0 = med(a0, Q0[x], Q0[x - 1]); p1 = med(a1, Q1[x], Q1[x - 1]); p2 = med(a2, Q2[x], Q2[x - 1]); }
                    U0[x] = zz(Y - p0); U1[x] = zz(U - p1); U2[x] = zz(V - p2);
                    a0 = Y; a1 = U; a2 = V;
                }
                for (int p = 0; p < 3; ++p) std::swap(prev[p], cur[p]);
            }
        }
    };

    // ---------------- token walk of one lane row (encoder and training share it) ----------------
    // f(tc, sym, mantissa, nm) ; state (k, c) is the lane state
    template <class F> inline void walk_row(const Tables& T, const uint16_t* row, int W, uint32_t gbase, uint32_t& k, uint32_t& c, F&& f) {
        uint32_t rl = 0;
        for (int x = 0; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            int rs, rm; uint32_t rman;
            if (rl < 256) { rs = T.renc[rl].rs; rm = T.renc[rl].rm; rman = T.renc[rl].rman; }
            else { rs = rs_of(rl); rm = rm_of_rs(rs); rman = rl - rbase_of_rs(rs); }
            const auto& vi = T.venc[v];
            f(gbase + ctx_of(k, c), rs * NV + vi.cv, (uint64_t(rman) << vi.vm) | vi.vman, rm + vi.vm);
            adapt(k, v); c = std::min(rl, 2u); rl = 0;
        }
        if (rl) f(gbase + ctx_of(k, c), EOB, 0, 0);
    }

    struct BitW {
        uint8_t* ptr; uint64_t acc = 0; int n = 0;
        inline void put(uint64_t v, int k) { acc = (acc << k) | v; n += k; if (n >= 32) { n -= 32; const uint32_t w = __builtin_bswap32(uint32_t(acc >> n)); std::memcpy(ptr, &w, 4); ptr += 4; } }
        inline uint8_t* flush() { while (n >= 8) { n -= 8; *ptr++ = uint8_t(acc >> n); } if (n > 0) { *ptr++ = uint8_t(acc << (8 - n)); n = 0; } acc = 0; return ptr; }
    };

    // encode the R8F1 body (no file header) of a W x H 8-bit RGB image
    inline std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: width must be 1..16383");
        const Tables& T = tables();
        std::vector<uint8_t> out; out.reserve(size_t(W) * H + 4096);
        Front fe(W);
        std::vector<uint16_t> bu(size_t(3) * 8 * W);
        const size_t LB = size_t(8) * W * 4 + 64;
        std::vector<uint8_t> lb[3]; for (auto& b : lb) b.resize(LB);
        uint32_t kr[3] = {16, 16, 16}, cr[3] = {0, 0, 0};
        for (int y0 = 0; y0 < H; y0 += 8) {
            const int nb = std::min(8, H - y0); const size_t PS = size_t(nb) * W;
            fe.band(rgb, y0, nb, bu.data());
            BitW w0{lb[0].data()}, w1{lb[1].data()}, w2{lb[2].data()};
            uint32_t k0 = kr[0], k1 = kr[1], k2 = kr[2], c0 = cr[0], c1 = cr[1], c2 = cr[2];
            for (int r = 0; r < nb; ++r) {      // the three lanes are independent: walk them one after another per row
                walk_row(T, bu.data() + size_t(r) * W, W, 0, k0, c0, [&](uint32_t tc, int s, uint64_t m, int nm) { const uint32_t cw = T.enc_cw[tc][s]; w0.put((uint64_t(cw & 0xFFFFu) << nm) | m, int(cw >> 16) + nm); });
                walk_row(T, bu.data() + PS + size_t(r) * W, W, NCTX, k1, c1, [&](uint32_t tc, int s, uint64_t m, int nm) { const uint32_t cw = T.enc_cw[tc][s]; w1.put((uint64_t(cw & 0xFFFFu) << nm) | m, int(cw >> 16) + nm); });
                walk_row(T, bu.data() + 2 * PS + size_t(r) * W, W, NCTX, k2, c2, [&](uint32_t tc, int s, uint64_t m, int nm) { const uint32_t cw = T.enc_cw[tc][s]; w2.put((uint64_t(cw & 0xFFFFu) << nm) | m, int(cw >> 16) + nm); });
            }
            kr[0] = k0; kr[1] = k1; kr[2] = k2; cr[0] = c0; cr[1] = c1; cr[2] = c2;
            const uint32_t ls[3] = {uint32_t(w0.flush() - lb[0].data()), uint32_t(w1.flush() - lb[1].data()), uint32_t(w2.flush() - lb[2].data())};
            const size_t o = out.size(); out.resize(o + 12 + ls[0] + ls[1] + ls[2]);
            uint8_t* d = out.data() + o; std::memcpy(d, ls, 12); d += 12;
            for (int p = 0; p < 3; ++p) { std::memcpy(d, lb[p].data(), ls[p]); d += ls[p]; }
        }
        return out;
    }

    // ---------------- decoder ----------------
    inline uint64_t peek(const uint8_t* base, uint64_t pos) { uint64_t w; std::memcpy(&w, base + (pos >> 3), 8); return __builtin_bswap64(w) << (pos & 7); }
    __attribute__((noinline, cold)) inline SlowSym decode_slow(const Tables& T, uint32_t tc, uint64_t w, bool& bad) {
        for (int l = 1; l <= MAXL; ++l) {
            const uint32_t v = uint32_t(w >> (64 - l)), f = T.first[tc][l];
            if (v - f < T.cntl[tc][l]) { SlowSym r = T.si[T.sorted[tc][T.offs[tc][l] + v - f]]; r.len = uint8_t(l); return r; }
        }
        bad = true; SlowSym r = T.si[EOB]; r.len = 1; return r;
    }
    // entropy pass of one band: u gets 3 * nb * W residuals (must be zeroed, plus 64 slack); returns false on a corrupt band
    __attribute__((noinline)) inline bool entropy_band(const Tables& T, const uint8_t* base, uint16_t* const u,
                                                       uint64_t p0, uint64_t p1, uint64_t p2, uint32_t st[3], uint32_t W, uint32_t nb) {
        const uint32_t PS = W * nb;
        uint32_t s0 = st[0], s1 = st[1], s2 = st[2];
        uint32_t q0 = 0, qe0 = W, q1 = PS, qe1 = PS + W, q2 = 2 * PS, qe2 = 2 * PS + W;
        const uint32_t e0 = PS, e1 = 2 * PS, e2 = 3 * PS, g1 = NCTX;
        const uint32_t* const lut = &T.lut[0][0]; bool bad = false;
#define RLH9_STEP(POS, ST, Q, QE, GB) do { \
        const uint64_t w_ = peek(base, POS); \
        const uint32_t kk_ = (ST & 255u) >> KR_SHIFT; \
        const uint32_t tc_ = (GB) + (kk_ > uint32_t(KR_MAX) ? uint32_t(KR_MAX) : kk_) * 3u + (ST >> 8); \
        const uint32_t e_ = lut[(size_t(tc_) << LUTB) | uint32_t(w_ >> (64 - LUTB))]; \
        if (__builtin_expect((e_ & 15u) != 0, 1)) { \
            const uint32_t len_ = e_ & 15u; \
            if (__builtin_expect(int32_t(e_) < 0, 0)) { POS += len_; Q = QE; } \
            else { \
                const uint32_t nm_ = (e_ >> 4) & 31u, vm_ = (e_ >> 9) & 7u; \
                const uint32_t m_ = uint32_t(((w_ << len_) >> 1) >> (63 - nm_)); \
                POS += len_ + nm_; \
                const uint32_t rl_ = ((e_ >> 22) & 63u) + (m_ >> vm_), v_ = ((e_ >> 12) & 1023u) + (m_ & ((1u << vm_) - 1u)); \
                Q += rl_; u[Q++] = uint16_t(v_); \
                uint32_t kr_ = ST & 255u; adapt(kr_, v_); ST = kr_ | (std::min(rl_, 2u) << 8); \
            } \
        } else { \
            const SlowSym ss_ = decode_slow(T, tc_, w_, bad); \
            const uint32_t m_ = uint32_t(((w_ << ss_.len) >> 1) >> (63 - ss_.nm)); \
            POS += ss_.len + ss_.nm; \
            if (ss_.eob) Q = QE; \
            else { \
                uint32_t rl_ = ss_.rbase + (m_ >> ss_.vm); const uint32_t v_ = ss_.vbase + (m_ & ss_.vmask); \
                if (rl_ >= QE - Q) { rl_ = QE - Q - 1; bad = true; } \
                Q += rl_; u[Q++] = uint16_t(v_); \
                uint32_t kr_ = ST & 255u; adapt(kr_, v_); ST = kr_ | (std::min(rl_, 2u) << 8); \
            } \
        } \
        if (__builtin_expect(Q >= QE, 0)) { bad |= Q > QE; Q = QE; QE += W; } \
    } while (0)
        while (q0 < e0 && q1 < e1 && q2 < e2) { RLH9_STEP(p0, s0, q0, qe0, 0); RLH9_STEP(p1, s1, q1, qe1, g1); RLH9_STEP(p2, s2, q2, qe2, g1); }
        while (q0 < e0) RLH9_STEP(p0, s0, q0, qe0, 0);
        while (q1 < e1) RLH9_STEP(p1, s1, q1, qe1, g1);
        while (q2 < e2) RLH9_STEP(p2, s2, q2, qe2, g1);
#undef RLH9_STEP
        st[0] = s0; st[1] = s1; st[2] = s2;
        return !bad;
    }
    // decode an R8F1 body of n bytes; the buffer must have 8 readable zero bytes after n plus slack (see decode_pad)
    inline size_t decode_pad(int W) { return size_t(W) * 8 * 3 * 6 + 64; }
    inline void decode(const uint8_t* base, size_t n, uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: bad size");
        const Tables& T = tables();
        std::vector<int16_t> prev[3], cur[3]; for (int p = 0; p < 3; ++p) { prev[p].assign(size_t(W), 0); cur[p].assign(size_t(W), 0); }
        std::vector<uint16_t> u(size_t(W) * 8 * 3 + 64, 0);
        size_t off = 0; uint32_t st[3] = {16, 16, 16};
        for (int y0 = 0; y0 < H; y0 += 8) {
            const int nb = std::min(8, H - y0);
            if (off + 12 > n) throw std::runtime_error("rlh9: truncated band header");
            uint32_t ls[3]; std::memcpy(ls, base + off, 12); off += 12;
            if (uint64_t(ls[0]) + ls[1] + ls[2] > n - off) throw std::runtime_error("rlh9: truncated band");
            const uint64_t p0 = uint64_t(off) * 8, p1 = p0 + uint64_t(ls[0]) * 8, p2 = p1 + uint64_t(ls[1]) * 8;
            off += size_t(ls[0]) + ls[1] + ls[2];
            const size_t PS = size_t(W) * nb;
            std::memset(u.data(), 0, PS * 3 * sizeof(uint16_t));
            if (!entropy_band(T, base, u.data(), p0, p1, p2, st, uint32_t(W), uint32_t(nb))) throw std::runtime_error("rlh9: corrupt band");
            for (int r = 0; r < nb; ++r) {
                const int y = y0 + r;
                const uint16_t *U0 = u.data() + size_t(r) * W, *U1 = U0 + PS, *U2 = U1 + PS;
                int16_t *C0 = cur[0].data(), *C1 = cur[1].data(), *C2 = cur[2].data();
                const int16_t *Q0 = prev[0].data(), *Q1 = prev[1].data(), *Q2 = prev[2].data();
                uint8_t* o = rgb + size_t(y) * W * 3;
                int a0, a1, a2;
                if (y == 0) { a0 = uz(U0[0]); a1 = uz(U1[0]); a2 = uz(U2[0]); }
                else { a0 = Q0[0] + uz(U0[0]); a1 = Q1[0] + uz(U1[0]); a2 = Q2[0] + uz(U2[0]); }
                C0[0] = int16_t(a0); C1[0] = int16_t(a1); C2[0] = int16_t(a2);
                { const int g = a0 - ((a1 + a2) >> 2); o[0] = uint8_t(a1 + g); o[1] = uint8_t(g); o[2] = uint8_t(a2 + g); }
                if (y == 0) {
                    for (int x = 1; x < W; ++x) {
                        a0 += uz(U0[x]); a1 += uz(U1[x]); a2 += uz(U2[x]);
                        C0[x] = int16_t(a0); C1[x] = int16_t(a1); C2[x] = int16_t(a2);
                        const int g = a0 - ((a1 + a2) >> 2); o[3 * x] = uint8_t(a1 + g); o[3 * x + 1] = uint8_t(g); o[3 * x + 2] = uint8_t(a2 + g);
                    }
                } else {
                    for (int x = 1; x < W; ++x) {
                        a0 = med(a0, Q0[x], Q0[x - 1]) + uz(U0[x]); a1 = med(a1, Q1[x], Q1[x - 1]) + uz(U1[x]); a2 = med(a2, Q2[x], Q2[x - 1]) + uz(U2[x]);
                        C0[x] = int16_t(a0); C1[x] = int16_t(a1); C2[x] = int16_t(a2);
                        const int g = a0 - ((a1 + a2) >> 2); o[3 * x] = uint8_t(a1 + g); o[3 * x + 1] = uint8_t(g); o[3 * x + 2] = uint8_t(a2 + g);
                    }
                }
                for (int p = 0; p < 3; ++p) std::swap(prev[p], cur[p]);
            }
        }
    }

    // ---------------- training (tools/rlh9) ----------------
    struct Counts { std::vector<double> c = std::vector<double>(size_t(TOTC) * NSYM, 0.0); };
    inline void count(Counts& C, const uint8_t* rgb, int W, int H) {
        static Tables* T0 = [] { auto* p = new Tables; for (int v = 1; v < 1024; ++v) { const int cv = cv_of(uint32_t(v)); p->venc[v] = {uint8_t(cv), uint8_t(vm_of_cv(cv)), uint16_t(uint32_t(v) - vbase_of_cv(cv))}; }
                                                     for (int r = 0; r < 256; ++r) { const int rs = rs_of(uint32_t(r)); p->renc[r] = {uint8_t(rs), uint8_t(rm_of_rs(rs)), uint16_t(uint32_t(r) - rbase_of_rs(rs))}; } return p; }();
        Front fe(W); std::vector<uint16_t> bu(size_t(3) * 8 * W); uint32_t kr[3] = {16, 16, 16}, cr[3] = {0, 0, 0};
        for (int y0 = 0; y0 < H; y0 += 8) {
            const int nb = std::min(8, H - y0); const size_t PS = size_t(nb) * W; fe.band(rgb, y0, nb, bu.data());
            for (int r = 0; r < nb; ++r) for (int p = 0; p < 3; ++p)
                walk_row(*T0, bu.data() + p * PS + size_t(r) * W, W, grp_of(p) * NCTX, kr[p], cr[p], [&](uint32_t tc, int s, uint64_t, int) { C.c[size_t(tc) * NSYM + s] += 1; });
        }
    }
    inline void huff_len(const double* c, int n, int lim, uint8_t* L) {
        std::vector<double> w(c, c + n); for (auto& x : w) x += 0.5;
        for (;;) {
            std::vector<int> par(size_t(2) * n, -1); using P = std::pair<double, int>; std::priority_queue<P, std::vector<P>, std::greater<P>> q;
            for (int i = 0; i < n; ++i) q.push({w[i], i}); int nx = n;
            while (q.size() > 1) { P a = q.top(); q.pop(); P b = q.top(); q.pop(); par[a.second] = nx; par[b.second] = nx; q.push({a.first + b.first, nx++}); }
            int mx = 0; for (int i = 0; i < n; ++i) { int d = 0; for (int j = i; par[j] >= 0; j = par[j]) ++d; L[i] = uint8_t(d); mx = std::max(mx, d); }
            if (mx <= lim) return;
            for (auto& x : w) x = x / 2 + 0.5;
        }
    }
}
