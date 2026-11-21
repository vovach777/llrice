// rlh9.hpp - LLRICE_ENGINE 9: RLH8 superscalar codec, format R8F4 (signature rev '9').
// Design: Argon (rlh8f.cpp, split micro-FSM, dual tables, R8F3 geometry, 2026-11-21/22) on top of Claude's RLH8s prototype.
//
// Front end: RCT (U = r-g, V = b-g, Y = g + ((U+V)>>2); RLH9_RCT 1: fNBLI), MED per plane (row 0: left,
//          column 0: top), zigzag. Chroma is coded as one plane S = zU + zV (0..2040) plus zU in 0..S as a truncated
//          code right after every non-zero S token (zV = S - zU).
// Geometry: 16-row bands; each band row of a plane is one chunk of width W; EOB = "rest of this row is zero".
// Lanes: 4 independent byte streams per band: Y rows 0..m-1, Y rows m..nb-1, S rows 0..m-1, S rows m..nb-1, m = nb / 2;
//          band header = 4 x u32 lane byte sizes; bit 31 of the first = screen table set.
// Alphabet (E8): 19 run classes (0..7 exact, 8..18 octave + mantissa) x 44 quarter-octave value classes + EOB = 837.
// Context: implicit, per lane, a split micro-automaton: sv (48 states, driven by the half-octave value class, calibrated
//          on llrice adapt_kr) and sr (16 states, an EMA 1/4 of the run bit width); ctx 0..47 per table set:
//          photos (sv >> 1) * 2 + (sr >> 3), screens (sv >> 2) * 4 + (sr >> 2).
//          The next state depends on the Huffman symbol only, so the decoder takes it from a byte table next to the LUT,
//          in parallel with the mantissa work. State (sv 12, sr 0) at the start of every lane of every band.
// Tables: 2 sets (photo, screen) x 2 groups (Y, S) x 48 = 192 static canonical Huffman tables (<= 15 bits); per set
//          and S = 1..15 a static code (<= 6 bits) for zU | S (RemLUT, Argon); S >= 16: truncated binary.
//          The encoder picks screen tables when >= RLH9_ZTHR % of the band's Y, U, V residuals are zero (encoder-only knob).
// Decoder: per band an entropy pass with the 4 lanes interleaved ONE TOKEN per lane per round (packed 32-bit LUT,
//          one LUT load + one class-byte load per token, the zU split read branch-free from the same 64-bit window),
//          then per 8 rows a SIMD wave MED (row k at column x - k, 8 rows = 8 int16 lanes of an SSE2 register)
//          and a vector inverse RCT. Scalar fallback without SSE2.
// Limits: 8-bit RGB only, width <= 16383 (run classes end at 16383).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>
#if defined(__SSE2__) && !defined(RLH9_NO_SIMD)
#include <immintrin.h>
#define RLH9_SSE 1
#endif
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
    constexpr int NCTX = 48, NGRP = 2, NSET = 2, SETC = NGRP * NCTX, TOTC = NSET * SETC;
#ifndef RLH9_ZTHR
#define RLH9_ZTHR 65   // percent of zero residuals in a band that selects the screen tables (encoder only)
#endif
    constexpr uint32_t MAX_W = 16383;
#ifndef RLH9_RCT
#define RLH9_RCT 0     // 0 (R8F3): U = R-G, V = B-G, Y = G + ((U+V)>>2); 1: fNBLI colour (V -= U>>2), needs other tables; with S = zU + zV
                       // the old RCT is better (photos -0.06%, 12 screens -0.4%, web8 -0.25%)
#endif
#ifndef RLH9_LUTB
#define RLH9_LUTB 9
#endif
    constexpr int LUTB = RLH9_LUTB;      // decoder-only knob, the format does not depend on it
#ifndef RLH9_NO_TABLES
    static_assert(RLH9_NSYM == NSYM && RLH9_TOTC == TOTC && RLH9_REMS == 15, "rlh9_tables.h does not match rlh9.hpp");
#endif

    constexpr int HB = 16, NLANE = 4;
    constexpr int REMS = 15, REMB = 6;   // zU | S for S <= REMS: static Huffman per set and S, codes <= REMB bits (RemLUT, Argon)
    // half-octave value class 0..15 (a function of the quarter-octave class only), run class min(bw(run), 7)
    inline int hoct_v(uint32_t v) { if (v <= 1) return 0; const int b = bw(v); if (b >= 9) return 15; return 2 * b - 3 + int((v >> (b - 2)) & 1u); }
    inline int rcls(uint32_t rl) { const int b = bw(rl); return b > 7 ? 7 : b; }
    // lane state st = sv | sr << 6
    constexpr uint32_t ST0 = 12;
// ctx (0..47) per table set: photos (sv >> 1) * 2 + (sr >> 3), screens (sv >> 2) * 4 + (sr >> 2) = (st & m) + ((st >> 6) >> c)
    inline uint32_t ctx_m(uint32_t set) { return set ? 60u : 62u; }
    inline uint32_t ctx_c(uint32_t set) { return set ? 2u : 3u; }
    inline uint32_t ctx_of(uint32_t st, uint32_t m, uint32_t c) { return (st & m) + ((st >> 6) >> c); }
    struct Fsm {
        uint8_t v[48][16], r[16][8];
        Fsm() {
            static const uint32_t rep_v[16] = {1, 2, 3, 4, 6, 9, 13, 19, 27, 39, 55, 79, 111, 159, 223, 384};
            for (int s = 0; s < 48; ++s) for (int b = 0; b < 16; ++b) {      // sv = kr * 2/3 of llrice adapt_kr
                uint32_t kr = uint32_t((s * 3 + 1) / 2); const uint32_t x = (rep_v[b] - 1) >> (kr >> 3);
                const uint32_t add = x >= 2 ? x : 0u, sub = x == 0 ? std::min(kr, 2u) : 0u;
                kr = std::min(kr + add, 240u) - sub; v[s][b] = uint8_t(std::min(int((kr * 2 + 1) / 3), 47));
            }
            // sr / 2.5 = EMA 1/4 of the run class: round((s/2.5 * 0.75 + b * 0.25) * 2.5), forced one step when the class
            // is 0.4 away; listed literally because the double rounding of the reference (r[14][0] = 10) is part of the format
            static const uint8_t R[16][8] = {{0,1,1,2,3,3,4,4},{1,2,2,3,3,4,5,5},{1,2,3,3,4,5,5,6},{2,3,4,4,5,5,6,7},{3,3,4,5,6,6,7,7},{4,4,5,6,6,7,8,8},{5,5,6,7,7,8,8,9},{5,6,6,7,8,8,9,10},
                                             {6,7,7,8,9,9,10,10},{7,7,8,8,9,10,11,11},{8,8,9,9,10,11,11,12},{8,9,10,10,11,12,12,13},{9,10,10,11,11,12,13,13},{10,10,11,12,12,13,14,14},{10,11,12,12,13,13,14,15},{11,12,13,13,14,14,15,15}};
            std::memcpy(r, R, sizeof r);
        }
    };
    inline const Fsm& fsm() { static const Fsm f; return f; }
    inline uint32_t next_st(const Fsm& F, uint32_t st, int vh, int rc) { return uint32_t(F.v[st & 63u][vh]) | (uint32_t(F.r[st >> 6][rc]) << 6); }
    inline int med(int a, int b, int c) { const int mx = a > b ? a : b, mn = a > b ? b : a; return std::max(mn, std::min(mx, a + b - c)); }
    inline uint16_t zz(int e) { return uint16_t((uint32_t(e) << 1) ^ uint32_t(e >> 31)); }
    inline int uz(uint32_t u) { return int(u >> 1) ^ -int(u & 1); }

    struct SlowSym { uint16_t rbase, vbase, vmask; uint8_t vm, nm, eob, len, vh, rc; };
    struct Tables {
        uint8_t len[TOTC][NSYM];
        uint32_t enc_cw[TOTC][NSYM];                     // code | len << 16
        uint32_t lut[TOTC][1 << LUTB];                   // len:4 nm:5 vm:3 vbase:11 rbase:6 ... eob:bit31; 0 = slow path
        uint8_t lut_cls[TOTC][1 << LUTB];                // rc << 4 | vh of the same entry: drives the state automaton
        uint16_t first[TOTC][MAXL + 2], cntl[TOTC][MAXL + 2], offs[TOTC][MAXL + 2], sorted[TOTC][NSYM];
        SlowSym si[NSYM];
        uint8_t rem_len[NSET][REMS + 1][REMS + 1];
        uint16_t rem_cw[NSET][REMS + 1][REMS + 1];        // code | len << 8
        uint16_t rem_lut[NSET][REMS + 1][1 << REMB];      // zU | len << 8 (row S = 0 unused)
        struct VEnc { uint8_t cv, vm, vh; uint16_t vman; } venc[2048];
        struct REnc { uint8_t rs, rm, rc; uint16_t rman; } renc[256];

        // rl: NSET x (REMS+1) x (REMS+1) code lengths of zU in 0..S (rows S = 1..REMS), each row a complete prefix code
        void build_rem(const uint8_t* rl) {
            std::memset(rem_cw, 0, sizeof rem_cw); std::memset(rem_lut, 0, sizeof rem_lut);
            for (int t = 0; t < NSET; ++t) for (int S = 1; S <= REMS; ++S) {
                uint32_t c = 0, kraft = 0;
                for (int z = 0; z <= S; ++z) { rem_len[t][S][z] = rl[(t * (REMS + 1) + S) * (REMS + 1) + z]; const int l = rem_len[t][S][z];
                    if (l < 1 || l > REMB) throw std::runtime_error("rlh9: bad rem table"); kraft += 1u << (REMB - l); }
                if (kraft != 1u << REMB) throw std::runtime_error("rlh9: incomplete rem table");
                for (int l = 1; l <= REMB; ++l) {
                    for (int z = 0; z <= S; ++z) if (rem_len[t][S][z] == l) {
                        rem_cw[t][S][z] = uint16_t(c | uint32_t(l) << 8);
                        for (uint32_t k = c << (REMB - l); k < (c + 1) << (REMB - l); ++k) rem_lut[t][S][k] = uint16_t(z | l << 8);
                        ++c;
                    }
                    c <<= 1;
                }
            }
        }
        void build(const uint8_t* packed_len) {          // packed_len: TOTC x ceil(NSYM/2) nibbles, low nibble first
            constexpr int PB = (NSYM + 1) / 2;
            for (int tc = 0; tc < TOTC; ++tc) for (int s = 0; s < NSYM; ++s) { const uint8_t b = packed_len[tc * PB + s / 2]; len[tc][s] = uint8_t(s & 1 ? b >> 4 : b & 15); }
            for (int s = 0; s < NSYM; ++s) {
                if (s == EOB) { si[s] = {0, 0, 0, 0, 0, 1, 0, 0, 0}; continue; }
                const int rs = s / NV, cv = s % NV; const uint8_t vm = uint8_t(vm_of_cv(cv));
                si[s] = {uint16_t(rbase_of_rs(rs)), uint16_t(vbase_of_cv(cv)), uint16_t((1u << vm) - 1u), vm, uint8_t(rm_of_rs(rs) + vm), 0, 0,
                         uint8_t(hoct_v(vbase_of_cv(cv))), uint8_t(rcls(rbase_of_rs(rs)))};
            }
            for (int tc = 0; tc < TOTC; ++tc) {
                const uint8_t* L = len[tc]; uint32_t c = 0, ns = 0;
                std::memset(lut[tc], 0, sizeof lut[tc]); std::memset(lut_cls[tc], 0, sizeof lut_cls[tc]);
                for (int l = 1; l <= MAXL; ++l) {
                    first[tc][l] = uint16_t(c); offs[tc][l] = uint16_t(ns); cntl[tc][l] = 0;
                    for (int s = 0; s < NSYM; ++s) if (L[s] == l) {
                        enc_cw[tc][s] = c | (uint32_t(l) << 16); sorted[tc][ns++] = uint16_t(s); ++cntl[tc][l];
                        if (l <= LUTB) {
                            uint32_t e = 0;
                            if (s == EOB) e = (1u << 31) | uint32_t(l);
                            else if (si[s].rbase < 64 && si[s].vbase < 2048 && si[s].vm <= 7)
                                e = uint32_t(l) | uint32_t(si[s].nm) << 4 | uint32_t(si[s].vm) << 9 | uint32_t(si[s].vbase) << 12 | uint32_t(si[s].rbase) << 23;
                            const uint8_t cl = uint8_t(si[s].rc << 4 | si[s].vh);
                            if (e) for (uint32_t k = c << (LUTB - l); k < (c + 1) << (LUTB - l); ++k) { lut[tc][k] = e; lut_cls[tc][k] = cl; }
                        }
                        ++c;
                    }
                    c <<= 1;
                }
                if (ns != uint32_t(NSYM)) throw std::runtime_error("rlh9: incomplete code table");
            }
            enc_luts();
        }
        void enc_luts() {
            for (int v = 1; v < 2048; ++v) { const int cv = cv_of(uint32_t(v)); venc[v] = {uint8_t(cv), uint8_t(vm_of_cv(cv)), uint8_t(hoct_v(vbase_of_cv(cv))), uint16_t(uint32_t(v) - vbase_of_cv(cv))}; }
            for (int r = 0; r < 256; ++r) { const int rs = rs_of(uint32_t(r)); renc[r] = {uint8_t(rs), uint8_t(rm_of_rs(rs)), uint8_t(rcls(rbase_of_rs(rs))), uint16_t(uint32_t(r) - rbase_of_rs(rs))}; }
        }
    };
#ifndef RLH9_NO_TABLES
    inline const Tables& tables() { static const Tables* t = [] { auto* p = new Tables; p->build(&RLH9_LEN[0][0]); p->build_rem(&RLH9_REMLEN[0][0][0]); return p; }(); return *t; }
#else
    const Tables& tables();
#endif

    // ---------------- front end: one band of residuals (zigzag), planes Y, U, V and S = zU + zV ----------------
    struct Front {
        int W; std::vector<int16_t> prev[3], cur[3];
        explicit Front(int w) : W(w) { for (int p = 0; p < 3; ++p) { prev[p].assign(size_t(w), 0); cur[p].assign(size_t(w), 0); } }
        // rgb: row-major 8-bit RGB, rows y0..y0+nb-1 -> u[p * nb * W + r * W + x], p = 0 Y, 1 U, 2 V, 3 S; returns the zero count of Y, U, V
        size_t band(const uint8_t* rgb, int y0, int nb, uint16_t* u) {
            const size_t PS = size_t(nb) * W; size_t z = 0;
            for (int r = 0; r < nb; ++r) {
                const int y = y0 + r; const uint8_t* s = rgb + size_t(y) * W * 3;
                int16_t *C0 = cur[0].data(), *C1 = cur[1].data(), *C2 = cur[2].data();
                const int16_t *Q0 = prev[0].data(), *Q1 = prev[1].data(), *Q2 = prev[2].data();
                uint16_t *U0 = u + size_t(r) * W, *U1 = U0 + PS, *U2 = U1 + PS, *US = U2 + PS;
                int a0 = 0, a1 = 0, a2 = 0;
                for (int x = 0; x < W; ++x) {
                    const int R = s[3 * x], G = s[3 * x + 1], B = s[3 * x + 2], U = R - G;
#if RLH9_RCT == 1
                    const int Y = G + ((U + (B - G) + 2) >> 2), V = (B - G) - (U >> 2);
#else
                    const int V = B - G, Y = G + ((U + V) >> 2);
#endif
                    C0[x] = int16_t(Y); C1[x] = int16_t(U); C2[x] = int16_t(V);
                    int p0, p1, p2;
                    if (y == 0) { p0 = a0; p1 = a1; p2 = a2; }
                    else if (x == 0) { p0 = Q0[0]; p1 = Q1[0]; p2 = Q2[0]; }
                    else { p0 = med(a0, Q0[x], Q0[x - 1]); p1 = med(a1, Q1[x], Q1[x - 1]); p2 = med(a2, Q2[x], Q2[x - 1]); }
                    const uint16_t z0 = zz(Y - p0), z1 = zz(U - p1), z2 = zz(V - p2);
                    U0[x] = z0; U1[x] = z1; U2[x] = z2; US[x] = uint16_t(z1 + z2);
                    z += size_t(z0 == 0) + size_t(z1 == 0) + size_t(z2 == 0);
                    a0 = Y; a1 = U; a2 = V;
                }
                for (int p = 0; p < 3; ++p) std::swap(prev[p], cur[p]);
            }
            return z;
        }
    };

    // ---------------- token walk of one lane row (encoder and training share it) ----------------
    // f(tc, sym, mantissa, nm, x) for every token (x = -1 for EOB); st is the lane state
    template <class F> inline void walk_row(const Tables& T, const uint16_t* row, int W, uint32_t gbase, uint32_t set, uint32_t& st, F&& f) {
        const Fsm& M = fsm(); uint32_t rl = 0; const uint32_t cm = ctx_m(set), cc = ctx_c(set);
        for (int x = 0; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            int rs, rm, rc; uint32_t rman;
            if (rl < 256) { rs = T.renc[rl].rs; rm = T.renc[rl].rm; rc = T.renc[rl].rc; rman = T.renc[rl].rman; }
            else { rs = rs_of(rl); rm = rm_of_rs(rs); rc = rcls(rbase_of_rs(rs)); rman = rl - rbase_of_rs(rs); }
            const auto& vi = T.venc[v];
            f(gbase + ctx_of(st, cm, cc), rs * NV + vi.cv, (uint64_t(rman) << vi.vm) | vi.vman, rm + vi.vm, x);
            st = next_st(M, st, vi.vh, rc); rl = 0;
        }
        if (rl) f(gbase + ctx_of(st, cm, cc), EOB, 0, 0, -1);
    }

    struct BitW {                                         // put() takes at most 32 bits at a time
        uint8_t* ptr; uint64_t acc = 0; int n = 0;
        inline void put(uint64_t v, int k) { acc = (acc << k) | v; n += k; if (n >= 32) { n -= 32; const uint32_t w = __builtin_bswap32(uint32_t(acc >> n)); std::memcpy(ptr, &w, 4); ptr += 4; } }
        // truncated binary code of x in 0..m-1, m >= 1
        inline void put_tbin(uint32_t x, uint32_t m) { const int k = bw(m) - 1; const uint32_t t = (2u << k) - m; if (x < t) put(x, k); else put(x + t, k + 1); }
        inline uint8_t* flush() { while (n >= 8) { n -= 8; *ptr++ = uint8_t(acc >> n); } if (n > 0) { *ptr++ = uint8_t(acc << (8 - n)); n = 0; } acc = 0; return ptr; }
    };

    // lane l of a band of nb rows: plane (0 = Y, 1 = S) and rows [r0, r1)
    inline void lane_rows(int l, int nb, int& plane, int& r0, int& r1) { const int m = nb / 2; plane = l >> 1; r0 = (l & 1) ? m : 0; r1 = (l & 1) ? nb : m; }

    // encode the R8F4 body (no file header) of a W x H 8-bit RGB image
    inline std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: width must be 1..16383");
        const Tables& T = tables();
        std::vector<uint8_t> out; out.reserve(size_t(W) * H + 4096);
        Front fe(W);
        std::vector<uint16_t> bu(size_t(4) * HB * W);
        const size_t LB = size_t(HB / 2) * W * 8 + 64;   // <= 47 bits per pixel token
        std::vector<uint8_t> lb[NLANE]; for (auto& b : lb) b.resize(LB);
        for (int y0 = 0; y0 < H; y0 += HB) {
            const int nb = std::min(HB, H - y0); const size_t PS = size_t(nb) * W;
            const size_t z = fe.band(rgb, y0, nb, bu.data());
            const uint32_t sc = uint32_t(z * 100 >= size_t(RLH9_ZTHR) * 3 * PS);
            uint32_t ls[NLANE];
            for (int l = 0; l < NLANE; ++l) {
                int plane, r0, r1; lane_rows(l, nb, plane, r0, r1);
                BitW w{lb[l].data()}; uint32_t st = ST0; const uint32_t gb = sc * SETC + uint32_t(plane) * NCTX;
                const uint16_t* P = bu.data() + (plane ? 3 * PS : 0); const uint16_t* PU = bu.data() + PS;
                for (int r = r0; r < r1; ++r) {
                    const uint16_t* row = P + size_t(r) * W; const uint16_t* rowu = PU + size_t(r) * W;
                    walk_row(T, row, W, gb, sc, st, [&](uint32_t tc, int s, uint64_t m, int nm, int x) {
                        const uint32_t cw = T.enc_cw[tc][s]; w.put(cw & 0xFFFFu, int(cw >> 16)); if (nm) w.put(m, nm);
                        if (plane && x >= 0) {
                            const uint32_t sv = row[x];
                            if (sv <= uint32_t(REMS)) { const uint32_t rc = T.rem_cw[sc][sv][rowu[x]]; w.put(rc & 255u, int(rc >> 8)); }
                            else w.put_tbin(rowu[x], sv + 1u);
                        }
                    });
                }
                ls[l] = uint32_t(w.flush() - lb[l].data());
            }
            if (ls[0] >= 0x80000000u) throw std::runtime_error("rlh9: band too large");
            const size_t o = out.size(); out.resize(o + 4 * NLANE + ls[0] + ls[1] + ls[2] + ls[3]);
            uint8_t* d = out.data() + o; const uint32_t hd[NLANE] = {ls[0] | sc << 31, ls[1], ls[2], ls[3]}; std::memcpy(d, hd, 4 * NLANE); d += 4 * NLANE;
            for (int l = 0; l < NLANE; ++l) { std::memcpy(d, lb[l].data(), ls[l]); d += ls[l]; }
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
    // entropy pass of one band: u gets Y, U, V residuals (3 * nb * W, must be zeroed, plus 128 slack); false on a corrupt band
    __attribute__((noinline)) inline bool entropy_band(const Tables& T, const uint8_t* base, uint16_t* const u,
                                                       const uint64_t pos[NLANE], uint32_t W, uint32_t nb, uint32_t sc) {
        const uint32_t PS = W * nb, MW = (nb / 2) * W;
        uint64_t p0 = pos[0], p1 = pos[1], p2 = pos[2], p3 = pos[3];
        uint32_t s0 = ST0, s1 = ST0, s2 = ST0, s3 = ST0;
        uint32_t q0 = 0, qe0 = W, q1 = MW, qe1 = MW + W, q2 = PS, qe2 = PS + W, q3 = PS + MW, qe3 = PS + MW + W;
        const uint32_t e0 = MW, e1 = PS, e2 = PS + MW, e3 = 2 * PS, gy = sc * SETC, gs = gy + NCTX, cm = ctx_m(sc), cc = ctx_c(sc);
        const uint32_t* const lut = &T.lut[0][0]; const uint8_t* const lcl = &T.lut_cls[0][0]; const uint16_t* const rlut = &T.rem_lut[sc][0][0]; bool bad = false;
        const Fsm& M = fsm();
#define RLH9_HEAD(POS, ST, GB) \
        const uint64_t w_ = peek(base, POS); \
        const uint32_t tc_ = (GB) + ctx_of(ST, cm, cc); \
        const size_t li_ = (size_t(tc_) << LUTB) | uint32_t(w_ >> (64 - LUTB)); \
        const uint32_t e_ = lut[li_]; const uint32_t cl_ = lcl[li_];
#define RLH9_TAIL(Q, QE) if (__builtin_expect(Q >= QE, 0)) { bad |= Q > QE; Q = QE; QE += W; }
#define RLH9_STEP_Y(POS, ST, Q, QE) do { RLH9_HEAD(POS, ST, gy) \
        if (__builtin_expect((e_ & 15u) != 0, 1)) { \
            const uint32_t len_ = e_ & 15u; \
            if (__builtin_expect(int32_t(e_) < 0, 0)) { POS += len_; Q = QE; } \
            else { \
                const uint32_t nm_ = (e_ >> 4) & 31u, vm_ = (e_ >> 9) & 7u; \
                const uint32_t m_ = uint32_t(((w_ << len_) >> 1) >> (63 - nm_)); \
                POS += len_ + nm_; \
                const uint32_t rl_ = ((e_ >> 23) & 63u) + (m_ >> vm_), v_ = ((e_ >> 12) & 2047u) + (m_ & ((1u << vm_) - 1u)); \
                Q += rl_; u[Q++] = uint16_t(v_); \
                ST = uint32_t(M.v[ST & 63u][cl_ & 15u]) | (uint32_t(M.r[ST >> 6][cl_ >> 4]) << 6); \
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
                ST = next_st(M, ST, ss_.vh, ss_.rc); \
            } \
        } \
        RLH9_TAIL(Q, QE) } while (0)
        // S lane: after the S token, zU from the same window (<= 9 + 20 + 11 bits on the fast path):
        // S <= REMS: RemLUT (6-bit index), else truncated binary of S + 1 values
#define RLH9_TBIN(WIN, SV, USED, POS, Q) \
            uint32_t uv_; \
            if (__builtin_expect((SV) <= uint32_t(REMS), 1)) { \
                const uint32_t re_ = rlut[((SV) << REMB) | uint32_t(((WIN) << (USED)) >> (64 - REMB))]; \
                POS += re_ >> 8; uv_ = re_ & 255u; \
            } else { \
                const uint32_t mm_ = (SV) + 1u, k_ = 31u - uint32_t(__builtin_clz(mm_)), t_ = (2u << k_) - mm_; \
                const uint32_t v2_ = uint32_t(((WIN) << (USED)) >> (63u - k_)), v1_ = v2_ >> 1, ge_ = uint32_t(v1_ >= t_); \
                POS += k_ + ge_; uv_ = ge_ ? v2_ - t_ : v1_; \
            } \
            u[Q] = uint16_t(uv_); u[Q + PS] = uint16_t((SV) - uv_); ++Q;
#define RLH9_STEP_S(POS, ST, Q, QE) do { RLH9_HEAD(POS, ST, gs) \
        if (__builtin_expect((e_ & 15u) != 0, 1)) { \
            const uint32_t len_ = e_ & 15u; \
            if (__builtin_expect(int32_t(e_) < 0, 0)) { POS += len_; Q = QE; } \
            else { \
                const uint32_t nm_ = (e_ >> 4) & 31u, vm_ = (e_ >> 9) & 7u; \
                const uint32_t m_ = uint32_t(((w_ << len_) >> 1) >> (63 - nm_)); \
                const uint32_t used_ = len_ + nm_; POS += used_; \
                const uint32_t rl_ = ((e_ >> 23) & 63u) + (m_ >> vm_), sv_ = ((e_ >> 12) & 2047u) + (m_ & ((1u << vm_) - 1u)); \
                Q += rl_; \
                RLH9_TBIN(w_, sv_, used_, POS, Q) \
                ST = uint32_t(M.v[ST & 63u][cl_ & 15u]) | (uint32_t(M.r[ST >> 6][cl_ >> 4]) << 6); \
            } \
        } else { \
            const SlowSym ss_ = decode_slow(T, tc_, w_, bad); \
            const uint32_t m_ = uint32_t(((w_ << ss_.len) >> 1) >> (63 - ss_.nm)); \
            POS += ss_.len + ss_.nm; \
            if (ss_.eob) Q = QE; \
            else { \
                uint32_t rl_ = ss_.rbase + (m_ >> ss_.vm); const uint32_t sv_ = ss_.vbase + (m_ & ss_.vmask); \
                if (rl_ >= QE - Q) { rl_ = QE - Q - 1; bad = true; } \
                Q += rl_; \
                const uint64_t wr_ = peek(base, POS); \
                RLH9_TBIN(wr_, sv_, 0u, POS, Q) \
                ST = next_st(M, ST, ss_.vh, ss_.rc); \
            } \
        } \
        RLH9_TAIL(Q, QE) } while (0)
        while (q0 < e0 && q1 < e1 && q2 < e2 && q3 < e3) { RLH9_STEP_Y(p0, s0, q0, qe0); RLH9_STEP_Y(p1, s1, q1, qe1); RLH9_STEP_S(p2, s2, q2, qe2); RLH9_STEP_S(p3, s3, q3, qe3); }
        while (q0 < e0) RLH9_STEP_Y(p0, s0, q0, qe0);
        while (q1 < e1) RLH9_STEP_Y(p1, s1, q1, qe1);
        while (q2 < e2) RLH9_STEP_S(p2, s2, q2, qe2);
        while (q3 < e3) RLH9_STEP_S(p3, s3, q3, qe3);
#undef RLH9_STEP_S
#undef RLH9_TBIN
#undef RLH9_STEP_Y
#undef RLH9_TAIL
#undef RLH9_HEAD
        return !bad;
    }

    // MED reconstruction of n <= 8 rows of one plane: zz residuals U (row stride W) -> P (row stride W), top = the row above
    // (unused when first: the image row 0 predicts from the left). Afterwards top = the last row.
    inline void recon_scalar(const uint16_t* U, int16_t* P, int16_t* top, int W, int n, bool first) {
        for (int k = 0; k < n; ++k) {
            const uint16_t* u = U + size_t(k) * W; int16_t* c = P + size_t(k) * W; const int16_t* q = k ? P + size_t(k - 1) * W : top;
            if (k == 0 && first) { int a = 0; for (int x = 0; x < W; ++x) { a += uz(u[x]); c[x] = int16_t(a); } continue; }
            int a = q[0] + uz(u[0]); c[0] = int16_t(a);
            for (int x = 1; x < W; ++x) { a = med(a, q[x], q[x - 1]) + uz(u[x]); c[x] = int16_t(a); }
        }
        std::memcpy(top, P + size_t(n - 1) * W, size_t(W) * sizeof(int16_t));
    }
#if defined(RLH9_SSE)
    inline void tr8x8(__m128i r[8]) {
        __m128i t0[8], t1[8];
        for (int i = 0; i < 8; i += 2) { t0[i] = _mm_unpacklo_epi16(r[i], r[i + 1]); t0[i + 1] = _mm_unpackhi_epi16(r[i], r[i + 1]); }
        for (int i = 0; i < 8; i += 4) {
            t1[i] = _mm_unpacklo_epi32(t0[i], t0[i + 2]); t1[i + 1] = _mm_unpackhi_epi32(t0[i], t0[i + 2]);
            t1[i + 2] = _mm_unpacklo_epi32(t0[i + 1], t0[i + 3]); t1[i + 3] = _mm_unpackhi_epi32(t0[i + 1], t0[i + 3]);
        }
        for (int i = 0; i < 4; ++i) { r[2 * i] = _mm_unpacklo_epi64(t1[i], t1[i + 4]); r[2 * i + 1] = _mm_unpackhi_epi64(t1[i], t1[i + 4]); }
    }
    // 8 rows as a wave: SSE lane k = row k at column d - k (Argon, after the PLS wave decoder). Same result as recon_scalar.
    __attribute__((noinline)) inline void recon_wave8(const uint16_t* U, int16_t* P, int16_t* top, int W, bool first) {
        if (W < 16) { recon_scalar(U, P, top, W, 8, first); return; }
        auto one = [&](int k, int x) {
            const int e = uz(U[size_t(k) * W + x]); int pr;
            if (k == 0 && first) pr = x ? P[x - 1] : 0;
            else { const int b = k ? P[size_t(k - 1) * W + x] : top[x]; pr = x ? med(P[size_t(k) * W + x - 1], b, k ? P[size_t(k - 1) * W + x - 1] : top[x - 1]) : b; }
            P[size_t(k) * W + x] = int16_t(pr + e);
        };
        for (int k = 0; k < 8; ++k) for (int x = 0; x <= 7 - k; ++x) one(k, x);
        alignas(16) int16_t a1[8], a2[8];
        for (int k = 0; k < 8; ++k) { a1[k] = P[size_t(k) * W + (7 - k)]; a2[k] = k <= 6 ? P[size_t(k) * W + (6 - k)] : 0; }
        __m128i prev1 = _mm_load_si128(reinterpret_cast<const __m128i*>(a1)), prev2 = _mm_load_si128(reinterpret_cast<const __m128i*>(a2));
        const __m128i one16 = _mm_set1_epi16(1), zero = _mm_setzero_si128();
        int d0 = 8;
        for (; d0 + 8 <= W; d0 += 8) {
            __m128i blk[8];
            for (int k = 0; k < 8; ++k) {
                const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U + size_t(k) * W + (d0 - k)));
                blk[k] = _mm_xor_si128(_mm_srli_epi16(v, 1), _mm_sub_epi16(zero, _mm_and_si128(v, one16)));
            }
            tr8x8(blk);
            for (int t = 0; t < 8; ++t) {
                const int d = d0 + t; const __m128i a = prev1;
                const __m128i b = _mm_insert_epi16(_mm_slli_si128(prev1, 2), first ? 0 : top[d], 0);
                const __m128i c = _mm_insert_epi16(_mm_slli_si128(prev2, 2), first ? 0 : top[d - 1], 0);
                __m128i pred = _mm_max_epi16(_mm_min_epi16(a, b), _mm_min_epi16(_mm_max_epi16(a, b), _mm_sub_epi16(_mm_add_epi16(a, b), c)));
                if (first) pred = _mm_insert_epi16(pred, _mm_extract_epi16(a, 0), 0);
                const __m128i cur = _mm_add_epi16(pred, blk[t]); blk[t] = cur; prev2 = prev1; prev1 = cur;
            }
            tr8x8(blk);
            for (int k = 0; k < 8; ++k) _mm_storeu_si128(reinterpret_cast<__m128i*>(P + size_t(k) * W + (d0 - k)), blk[k]);
        }
        for (int k = 0; k < 8; ++k) for (int x = d0 - k; x < W; ++x) one(k, x);
        std::memcpy(top, P + size_t(7) * W, size_t(W) * sizeof(int16_t));
    }
#endif
    inline void irct_row(const int16_t* Y, const int16_t* U, const int16_t* V, uint8_t* o, int W) {
        int x = 0;
#if defined(RLH9_SSE) && defined(__SSSE3__)
        const __m128i mR0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5), mG0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1),
                      mB0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1), mR1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1),
                      mG1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10), mB1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1),
                      mR2 = _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1), mG2 = _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1),
                      mB2 = _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15);
        const __m128i k2 = _mm_set1_epi16(2), m255 = _mm_set1_epi16(255);
        for (; x + 16 <= W; x += 16) {
            __m128i r8[2], g8[2], b8[2];
            for (int h = 0; h < 2; ++h) {
                const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Y + x + 8 * h)), uu = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U + x + 8 * h));
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(V + x + 8 * h));
#if RLH9_RCT == 1
                v = _mm_add_epi16(v, _mm_srai_epi16(uu, 2));
                const __m128i g = _mm_sub_epi16(y, _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(uu, v), k2), 2));
#else
                const __m128i g = _mm_sub_epi16(y, _mm_srai_epi16(_mm_add_epi16(uu, v), 2));
#endif
                r8[h] = _mm_and_si128(_mm_add_epi16(uu, g), m255); g8[h] = _mm_and_si128(g, m255); b8[h] = _mm_and_si128(_mm_add_epi16(v, g), m255);
            }
            const __m128i r = _mm_packus_epi16(r8[0], r8[1]), g = _mm_packus_epi16(g8[0], g8[1]), b = _mm_packus_epi16(b8[0], b8[1]);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR0), _mm_shuffle_epi8(g, mG0)), _mm_shuffle_epi8(b, mB0)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 16), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR1), _mm_shuffle_epi8(g, mG1)), _mm_shuffle_epi8(b, mB1)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 32), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR2), _mm_shuffle_epi8(g, mG2)), _mm_shuffle_epi8(b, mB2)));
        }
#endif
        for (; x < W; ++x) {
            const int y = Y[x], uu = U[x];
#if RLH9_RCT == 1
            const int v = V[x] + (uu >> 2), g = y - ((uu + v + 2) >> 2);
#else
            const int v = V[x], g = y - ((uu + v) >> 2);
#endif
            o[3 * x] = uint8_t(uu + g); o[3 * x + 1] = uint8_t(g); o[3 * x + 2] = uint8_t(v + g);
        }
    }

    // decode an R8F4 body of n bytes; the buffer must have 8 readable zero bytes after n plus slack (see decode_pad)
    inline size_t decode_pad(int W) { return size_t(W) * HB * 2 * 6 + 64; }
    inline void decode(const uint8_t* base, size_t n, uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: bad size");
        const Tables& T = tables();
        std::vector<int16_t> top[3], P[3]; for (int p = 0; p < 3; ++p) { top[p].assign(size_t(W), 0); P[p].assign(size_t(8) * W, 0); }
        std::vector<uint16_t> u(size_t(W) * HB * 3 + 128, 0);
        size_t off = 0;
        for (int y0 = 0; y0 < H; y0 += HB) {
            const int nb = std::min(HB, H - y0);
            if (off + 4 * NLANE > n) throw std::runtime_error("rlh9: truncated band header");
            uint32_t ls[NLANE]; std::memcpy(ls, base + off, 4 * NLANE); off += 4 * NLANE;
            const uint32_t sc = ls[0] >> 31; ls[0] &= 0x7FFFFFFFu;
            if (uint64_t(ls[0]) + ls[1] + ls[2] + ls[3] > n - off) throw std::runtime_error("rlh9: truncated band");
            uint64_t pos[NLANE]; for (int l = 0; l < NLANE; ++l) { pos[l] = uint64_t(off) * 8; off += ls[l]; }
            const size_t PS = size_t(W) * nb;
            std::memset(u.data(), 0, PS * 3 * sizeof(uint16_t));
            if (!entropy_band(T, base, u.data(), pos, uint32_t(W), uint32_t(nb), sc)) throw std::runtime_error("rlh9: corrupt band");
            for (int sub = 0; sub < nb; sub += 8) {
                const int sn = std::min(8, nb - sub); const bool first = y0 + sub == 0;
                for (int p = 0; p < 3; ++p) {
                    const uint16_t* U = u.data() + size_t(p) * PS + size_t(sub) * W;
#if defined(RLH9_SSE)
                    if (sn == 8) recon_wave8(U, P[p].data(), top[p].data(), W, first); else
#endif
                    recon_scalar(U, P[p].data(), top[p].data(), W, sn, first);
                }
                for (int r = 0; r < sn; ++r)
                    irct_row(P[0].data() + size_t(r) * W, P[1].data() + size_t(r) * W, P[2].data() + size_t(r) * W, rgb + size_t(y0 + sub + r) * W * 3, W);
            }
        }
    }

    // ---------------- training (tools/rlh9) ----------------
    struct Counts { std::vector<double> c = std::vector<double>(size_t(TOTC) * NSYM, 0.0), rem = std::vector<double>(size_t(NSET) * (REMS + 1) * (REMS + 1), 0.0); };
    // counts per (set, group, ctx, sym); force < 0: band class by the encoder detector, else that set; wt: weight per token
    inline void count(Counts& C, const uint8_t* rgb, int W, int H, int force = -1, double wt = 1.0, uint64_t* nband = nullptr) {
        static Tables* T0 = [] { auto* p = new Tables; p->enc_luts(); return p; }();
        Front fe(W); std::vector<uint16_t> bu(size_t(4) * HB * W);
        for (int y0 = 0; y0 < H; y0 += HB) {
            const int nb = std::min(HB, H - y0); const size_t PS = size_t(nb) * W;
            const size_t z = fe.band(rgb, y0, nb, bu.data());
            const uint32_t sc = force < 0 ? uint32_t(z * 100 >= size_t(RLH9_ZTHR) * 3 * PS) : uint32_t(force);
            if (nband) ++nband[sc];
            for (int l = 0; l < NLANE; ++l) {
                int plane, r0, r1; lane_rows(l, nb, plane, r0, r1); uint32_t st = ST0;
                const uint16_t* Pl = bu.data() + (plane ? 3 * PS : 0);
                for (int r = r0; r < r1; ++r) {
                    const uint16_t* row = Pl + size_t(r) * W; const uint16_t* rowu = bu.data() + PS + size_t(r) * W;
                    walk_row(*T0, row, W, sc * SETC + uint32_t(plane) * NCTX, sc, st, [&](uint32_t tc, int s, uint64_t, int, int x) {
                        C.c[size_t(tc) * NSYM + s] += wt;
                        if (plane && x >= 0 && row[x] <= REMS) C.rem[(size_t(sc) * (REMS + 1) + row[x]) * (REMS + 1) + rowu[x]] += wt;
                    });
                }
            }
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
