// rlh9.hpp - LLRICE_ENGINE 9: RLH8 superscalar codec, format R8F4 (signature rev '9').
// Design: Argon (rlh8f.cpp, split micro-FSM, dual tables, R8F3 geometry, 2026-11-21/22) on top of Claude's RLH8s prototype.
// Fast encoder (V3 + ablations, 2026-11-22): fused SIMD front (AVX2 / SSE2+SSSE3 / scalar), bitmask zero-skip (tzcnt),
// branchless 64-bit BitW_Fast (single put per token), optional 2-lane superscalar & branchless zU|S.
// Byte-identical output to 01a4303 scalar rlh9::encode across AVX2, SSE2/SSSE3, and RLH9_NO_SIMD builds.
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

// Encoder ablation knobs (default: V3_BEST; format and output bytes are 100% invariant)
// RLH9_ENC_MODE:
//   0 = original 01a4303 scalar encoder (reference)
//   1 = V3 (fused SIMD front + bitmask tzcnt + branchless BitW_Fast single-put + branchless tbin)
//   2 = V3 + Idea 1a (2-lane interleaved: Y0+Y1, then S0+S1)
//   3 = V3 + Idea 1b (2-lane interleaved: Y0+S0, then Y1+S1)
//   4 = V3 + Idea 2a (bitmask tzcnt only in screen bands sc==1; simple scan in photo bands sc==0)
//   5 = V3 + Idea 2b (bitmask tzcnt when band zero-ratio >= RLH9_MASK_THR%, else simple scan)
//   6 = V3 + Idea 3  (branchless cmov for zU|S: rem_cw vs tbin_tbl[2048], no S<=15 branch)
//   7 = V3 + Idea 2a + Idea 1b (combined test)
//   8 = V3 + Idea 2a + Idea 3  (combined test)
#ifndef RLH9_ENC_MODE
#define RLH9_ENC_MODE 5
#endif
#ifndef RLH9_MASK_THR
#define RLH9_MASK_THR 30
#endif
// LUT vs ALU knobs (apply to modes >= 1):
// RLH9_VENC_MODE: 0 = original struct VEnc[2048], 1 = packed uint32_t venc32[2048], 2 = pure ALU (clz + shifts)
#ifndef RLH9_VENC_MODE
#define RLH9_VENC_MODE 0
#endif
// RLH9_RENC_MODE: 0 = original struct REnc[256], 1 = packed uint32_t renc32[256], 2 = pure ALU (clz + shifts)
#ifndef RLH9_RENC_MODE
#define RLH9_RENC_MODE 0
#endif
// RLH9_TBIN_MODE: 0 = ALU clz (with S<=15 branch), 1 = tbin_tbl[2048] LUT (with S<=15 branch), 2 = tbin_tbl[2048] branchless cmov
#ifndef RLH9_TBIN_MODE
#define RLH9_TBIN_MODE 1
#endif
// RLH9_FSM_ALU: 0 = Fsm::v[48][16] LUT, 1 = ALU adapt_kr for sv (sr always 128B LUT due to double-rounding format rule)
#ifndef RLH9_FSM_ALU
#define RLH9_FSM_ALU 0
#endif

// Decoder ablation knobs (format and decoded pixels are 100% invariant)
// RLH9_DEC_NO_MEMSET: 0 = memset(u, 0), 1 = write zeros on-the-fly during entropy_band
#ifndef RLH9_DEC_NO_MEMSET
#define RLH9_DEC_NO_MEMSET 0
#endif
// RLH9_DEC_2L_DRAIN: 0 = 4 single-lane drains, 1 = 2-lane Y0+Y1 and S0+S1 drains before 1-lane drains
#ifndef RLH9_DEC_2L_DRAIN
#define RLH9_DEC_2L_DRAIN 1
#endif
// RLH9_DEC_LUT64: 0 = uint32_t lut + uint8_t lut_cls (2 loads), 1 = single uint64_t lut64 load (packed cls/vmask/sh)
#ifndef RLH9_DEC_LUT64
#define RLH9_DEC_LUT64 0
#endif
// RLH9_DEC_SLOW_MODE: 0 = original loop decode_slow, 1 = 2-level O(1) sub_lut (64-entry blocks), 2 = SIMD 15-threshold canonical search
#ifndef RLH9_DEC_SLOW_MODE
#define RLH9_DEC_SLOW_MODE 1
#endif
// RLH9_DEC_TBIN_MODE: 0 = ALU clz (original), 1 = tbin_tbl[2048] LUT + S<=15 branch, 2 = tbin_tbl[2048] branchless cmov
#ifndef RLH9_DEC_TBIN_MODE
#define RLH9_DEC_TBIN_MODE 1
#endif
// RLH9_DEC_WAVE_MODE: 0 = original SSE2 1-plane wave, 1 = AVX2 2-plane (U+V) wave + Y wave, 2 = fused 3-plane wave + iRCT + RGB24 (no P buffer!)
#ifndef RLH9_DEC_WAVE_MODE
#define RLH9_DEC_WAVE_MODE 3
#endif
// RLH9_DEC_IRCT_AVX2: 0 = SSSE3 16px irct_row, 1 = AVX2 32px irct_row (used when WAVE_MODE != 2)
#ifndef RLH9_DEC_IRCT_AVX2
#define RLH9_DEC_IRCT_AVX2 1
#endif


namespace rlh9
{
    inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
    inline int cv_of(uint32_t v) { const int b = bw(v); if (b <= 1) return b; if (b == 2) return 4 + int(v & 1); return 4 * b - 4 + int((v >> (b - 3)) & 3); }
    inline int vm_of_cv(int cv) { return cv <= 7 ? 0 : (cv + 4) / 4 - 3; }
    inline uint32_t vbase_of_cv(int cv) { if (cv <= 1) return uint32_t(cv); if (cv == 4 || cv == 5) return uint32_t(cv - 2); if (cv < 8) return 1; int b = (cv + 4) / 4, q = (cv + 4) % 4; return (1u << (b - 1)) | (uint32_t(q) << (b - 3)); }
    inline int rs_of(uint32_t rl) { return rl < 8 ? int(rl) : 8 + bw(rl) - 4; }
    inline int rm_of_rs(int rs) { return rs < 8 ? 0 : rs - 8 + 3; }
    inline uint32_t rbase_of_rs(int rs) { return rs < 8 ? uint32_t(rs) : (1u << (rs - 8 + 3)); }

    constexpr int NV = 44, NR = 19, NSYM = NR * NV + 1, EOB = NSYM - 1, MAXL = 15;
    constexpr int NCTX = 48, NGRP = 2, NSET = 2, SETC = NGRP * NCTX, TOTC = NSET * SETC;
#ifndef RLH9_ZTHR
#define RLH9_ZTHR 65
#endif
    constexpr uint32_t MAX_W = 16383;
#ifndef RLH9_RCT
#define RLH9_RCT 0
#endif
#ifndef RLH9_LUTB
#define RLH9_LUTB 9
#endif
    constexpr int LUTB = RLH9_LUTB;
#ifndef RLH9_NO_TABLES
    static_assert(RLH9_NSYM == NSYM && RLH9_TOTC == TOTC && RLH9_REMS == 15, "rlh9_tables.h does not match rlh9.hpp");
#endif

    constexpr int HB = 16, NLANE = 4;
    constexpr int REMS = 15, REMB = 6;
    inline int hoct_v(uint32_t v) { if (v <= 1) return 0; const int b = bw(v); if (b >= 9) return 15; return 2 * b - 3 + int((v >> (b - 2)) & 1u); }
    inline int rcls(uint32_t rl) { const int b = bw(rl); return b > 7 ? 7 : b; }
    constexpr uint32_t ST0 = 12;
    inline uint32_t ctx_m(uint32_t set) { return set ? 60u : 62u; }
    inline uint32_t ctx_c(uint32_t set) { return set ? 2u : 3u; }
    inline uint32_t ctx_of(uint32_t st, uint32_t m, uint32_t c) { return (st & m) + ((st >> 6) >> c); }
    struct Fsm {
        uint8_t v[48][16], r[16][8]; uint16_t r6[16][8];
        Fsm() {
            static const uint32_t rep_v[16] = {1, 2, 3, 4, 6, 9, 13, 19, 27, 39, 55, 79, 111, 159, 223, 384};
            for (int s = 0; s < 48; ++s) for (int b = 0; b < 16; ++b) {
                uint32_t kr = uint32_t((s * 3 + 1) / 2); const uint32_t x = (rep_v[b] - 1) >> (kr >> 3);
                const uint32_t add = x >= 2 ? x : 0u, sub = x == 0 ? std::min(kr, 2u) : 0u;
                kr = std::min(kr + add, 240u) - sub; v[s][b] = uint8_t(std::min(int((kr * 2 + 1) / 3), 47));
            }
            static const uint8_t R[16][8] = {{0,1,1,2,3,3,4,4},{1,2,2,3,3,4,5,5},{1,2,3,3,4,5,5,6},{2,3,4,4,5,5,6,7},{3,3,4,5,6,6,7,7},{4,4,5,6,6,7,8,8},{5,5,6,7,7,8,8,9},{5,6,6,7,8,8,9,10},
                                             {6,7,7,8,9,9,10,10},{7,7,8,8,9,10,11,11},{8,8,9,9,10,11,11,12},{8,9,10,10,11,12,12,13},{9,10,10,11,11,12,13,13},{10,10,11,12,12,13,14,14},{10,11,12,12,13,13,14,15},{11,12,13,13,14,14,15,15}};
            std::memcpy(r, R, sizeof r); for (int s = 0; s < 16; ++s) for (int b = 0; b < 8; ++b) r6[s][b] = uint16_t(R[s][b]) << 6;
        }
    };
    inline const Fsm& fsm() { static const Fsm f; return f; }
    inline uint32_t next_st_alu_v(uint32_t sv, int vh) {
        static const uint32_t rep_v[16] = {1, 2, 3, 4, 6, 9, 13, 19, 27, 39, 55, 79, 111, 159, 223, 384};
        uint32_t kr = (sv * 3u + 1u) >> 1;
        const uint32_t x = (rep_v[vh] - 1u) >> (kr >> 3);
        const uint32_t add = x >= 2u ? x : 0u;
        const uint32_t sub = x == 0u ? (kr < 2u ? kr : 2u) : 0u;
        kr = (kr + add > 240u ? 240u : kr + add) - sub;
        const uint32_t nv = (kr * 2u + 1u) / 3u;
        return nv > 47u ? 47u : nv;
    }
    inline uint32_t next_st(const Fsm& F, uint32_t st, int vh, int rc) {
#if RLH9_FSM_ALU == 1
        return next_st_alu_v(st & 63u, vh) | (uint32_t(F.r[st >> 6][rc]) << 6);
#else
        return uint32_t(F.v[st & 63u][vh]) | (uint32_t(F.r[st >> 6][rc]) << 6);
#endif
    }
    inline int med(int a, int b, int c) { const int mx = a > b ? a : b, mn = a > b ? b : a; return std::max(mn, std::min(mx, a + b - c)); }
    inline uint16_t zz(int e) { return uint16_t((uint32_t(e) << 1) ^ uint32_t(e >> 31)); }
    inline int uz(uint32_t u) { return int(u >> 1) ^ -int(u & 1); }

    struct SlowSym { uint16_t rbase, vbase, vmask; uint8_t vm, nm, eob, len, vh, rc; };
    struct Tables {
        uint8_t len[TOTC][NSYM];
        uint32_t enc_cw[TOTC][NSYM];                     // code | len << 16
        uint32_t lut[TOTC][1 << LUTB];
        uint8_t lut_cls[TOTC][1 << LUTB];
        uint16_t first[TOTC][MAXL + 2], cntl[TOTC][MAXL + 2], offs[TOTC][MAXL + 2], sorted[TOTC][NSYM];
        SlowSym si[NSYM];
        uint8_t rem_len[NSET][REMS + 1][REMS + 1];
        uint16_t rem_cw[NSET][REMS + 1][REMS + 1];        // code | len << 8
        uint16_t rem_lut[NSET][REMS + 1][1 << REMB];
        struct VEnc { uint8_t cv, vm, vh; uint16_t vman; } venc[2048];
        struct REnc { uint8_t rs, rm, rc; uint16_t rman; } renc[256];
        // Compact 32-bit packed encoder tables (1 load per lookup)
        // venc32[v]: cv:6 | vm:4 << 6 | vh:4 << 10 | vman:11 << 14
        // renc32[r]: rs:5 | rm:4 << 5 | rc:3 << 9  | rman:14 << 12
        // tbin_tbl[sv]: k:4 | t:12 << 4  (for sv = 1..2047, m = sv + 1)
        uint32_t venc32[2048];
        uint32_t renc32[256];
        uint16_t tbin_tbl[2048];
        // Fast decoder auxiliary tables:
#if RLH9_DEC_LUT64 == 1
        uint64_t lut64[TOTC][1 << LUTB];
#endif
        std::vector<uint16_t> sub_lut;                   // 2-level O(1) slow-path table: sym | (len << 10)
        alignas(32) int16_t lim16[TOTC][16];             // biased 16-bit thresholds for SIMD canonical search
        uint16_t lim_true_mask[TOTC];                    // bitmask of lanes (0..13) where thresh == 65536

        void build_rem(const uint8_t* rl) {
            std::memset(rem_cw, 0, sizeof rem_cw); std::memset(rem_lut, 0, sizeof rem_lut);
            for (int t = 0; t < NSET; ++t) for (int S = 1; S <= REMS; ++S) {
                uint32_t c = 0, kraft = 0;
                for (int z = 0; z <= S; ++z) { rem_len[t][S][z] = rl[(t * (REMS + 1) + S) * (REMS + 1) + z]; const int l = rem_len[t][S][z];
                    if (l < 1 || l > REMB) throw std::runtime_error("rlh9: bad rem table");
                    kraft += 1u << (REMB - l); }
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
        void build(const uint8_t* packed_len) {
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
                // Build 64-bit packed LUT, 2-level O(1) slow sub_lut, and SIMD 15-threshold lim16
#if RLH9_DEC_LUT64 == 1
                std::memset(lut64[tc], 0, sizeof lut64[tc]);
                for (int k = 0; k < (1 << LUTB); ++k) {
                    const uint32_t e = lut[tc][k];
                    if (e) {
                        const uint32_t cl = lut_cls[tc][k];
                        const uint32_t nm = (e >> 4) & 31u, vm = (e >> 9) & 7u;
                        const uint64_t vh = cl & 15u, rc = cl >> 4;
                        const uint64_t vmask = (1u << vm) - 1u, sh = 63u - nm;
                        lut64[tc][k] = uint64_t(e) | (vh << 32) | (rc << 36) | (vmask << 40) | (sh << 48);
                    }
                }
#endif
                // Allocate 64-entry sub_lut blocks for every slow 9-bit prefix (lut[tc][k] == 0)
                // Store block_idx << 4 in lut[tc][k] and lut64[tc][k] (bits 0..3 stay 0 => slow path!)
                constexpr int SUBB = MAXL - LUTB; // 6 bits -> 64 entries
                for (int k = 0; k < (1 << LUTB); ++k) {
                    if (lut[tc][k] == 0) {
                        const uint32_t bidx = uint32_t(sub_lut.size() >> SUBB);
                        sub_lut.resize(sub_lut.size() + (1u << SUBB), 0);
                        lut[tc][k] = bidx << 4;
#if RLH9_DEC_LUT64 == 1
                        lut64[tc][k] = uint64_t(bidx) << 4;
#endif
                    }
                }
                // Fill sub_lut blocks for all slow symbols (either l > LUTB, or l <= LUTB with e == 0)
                for (int l = 1; l <= MAXL; ++l) {
                    const uint32_t cnt = cntl[tc][l], f0 = first[tc][l], o0 = offs[tc][l];
                    for (uint32_t j = 0; j < cnt; ++j) {
                        const uint32_t code = f0 + j;
                        const uint16_t sym = sorted[tc][o0 + j];
                        const uint16_t val = uint16_t(sym | (uint32_t(l) << 10));
                        if (l <= LUTB) {
                            const uint32_t k0 = code << (LUTB - l), k1 = (code + 1u) << (LUTB - l);
                            for (uint32_t k = k0; k < k1; ++k) {
                                if ((lut[tc][k] & 15u) == 0) {
                                    const size_t base = size_t(lut[tc][k] >> 4) << SUBB;
                                    for (size_t sfx = 0; sfx < (1u << SUBB); ++sfx) sub_lut[base + sfx] = val;
                                }
                            }
                        } else {
                            const uint32_t k = code >> (l - LUTB);
                            const uint32_t rem_bits = code & ((1u << (l - LUTB)) - 1u);
                            const uint32_t s0 = rem_bits << (MAXL - l), s1 = (rem_bits + 1u) << (MAXL - l);
                            const size_t base = size_t(lut[tc][k] >> 4) << SUBB;
                            for (uint32_t sfx = s0; sfx < s1; ++sfx) sub_lut[base + sfx] = val;
                        }
                    }
                }
                // Build SIMD 15-threshold lim16 for l = 1..14 (lane 14,15 padded)
                uint16_t tmask = 0;
                for (int l = 1; l <= 14; ++l) {
                    const uint32_t end_l = (uint32_t(first[tc][l]) + uint32_t(cntl[tc][l])) << (16 - l);
                    if (end_l >= 65536u) { lim16[tc][l - 1] = 32767; tmask |= uint16_t(1u << (l - 1)); }
                    else lim16[tc][l - 1] = int16_t(end_l ^ 0x8000u);
                }
                lim16[tc][14] = 32767; lim16[tc][15] = 32767;
                lim_true_mask[tc] = tmask;
            }
            enc_luts();
        }
        void enc_luts() {
            venc32[0] = 0; tbin_tbl[0] = 0;
            for (int v = 1; v < 2048; ++v) {
                const int cv = cv_of(uint32_t(v));
                const int vm = vm_of_cv(cv);
                const int vh = hoct_v(vbase_of_cv(cv));
                const uint32_t vman = uint32_t(v) - vbase_of_cv(cv);
                venc[v] = {uint8_t(cv), uint8_t(vm), uint8_t(vh), uint16_t(vman)};
                venc32[v] = uint32_t(cv) | (uint32_t(vm) << 6) | (uint32_t(vh) << 10) | (vman << 14);
                const uint32_t m = uint32_t(v) + 1u;
                const uint32_t k = 31u - uint32_t(__builtin_clz(m));
                const uint32_t t = (2u << k) - m;
                tbin_tbl[v] = uint16_t(k | (t << 4));
            }
            for (int r = 0; r < 256; ++r) {
                const int rs = rs_of(uint32_t(r));
                const int rm = rm_of_rs(rs);
                const int rc = rcls(rbase_of_rs(rs));
                const uint32_t rman = uint32_t(r) - rbase_of_rs(rs);
                renc[r] = {uint8_t(rs), uint8_t(rm), uint8_t(rc), uint16_t(rman)};
                renc32[r] = uint32_t(rs) | (uint32_t(rm) << 5) | (uint32_t(rc) << 9) | (rman << 12);
            }
        }
    };
#ifndef RLH9_NO_TABLES
    inline const Tables& tables() { static const Tables* t = [] { auto* p = new Tables; p->build(&RLH9_LEN[0][0]); p->build_rem(&RLH9_REMLEN[0][0][0]); return p; }(); return *t; }
#else
    const Tables& tables();
#endif

    // ---------------- front end: one band of residuals (zigzag), planes Y, U, V and S = zU + zV ----------------
#if defined(RLH9_SSE) && defined(__SSSE3__)
    inline void rgb24_deint16_ssse3(const uint8_t* p, __m128i& r8, __m128i& g8, __m128i& b8) {
        const __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
        const __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));
        const __m128i mR0 = _mm_setr_epi8(0,3,6,9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        const __m128i mR1 = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,2,5,8,11,14,-1,-1,-1,-1,-1);
        const __m128i mR2 = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,4,7,10,13);
        const __m128i mG0 = _mm_setr_epi8(1,4,7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        const __m128i mG1 = _mm_setr_epi8(-1,-1,-1,-1,-1,0,3,6,9,12,15,-1,-1,-1,-1,-1);
        const __m128i mG2 = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,5,8,11,14);
        const __m128i mB0 = _mm_setr_epi8(2,5,8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        const __m128i mB1 = _mm_setr_epi8(-1,-1,-1,-1,-1,1,4,7,10,13,-1,-1,-1,-1,-1,-1);
        const __m128i mB2 = _mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,3,6,9,12,15);
        r8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, mR0), _mm_shuffle_epi8(c1, mR1)), _mm_shuffle_epi8(c2, mR2));
        g8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, mG0), _mm_shuffle_epi8(c1, mG1)), _mm_shuffle_epi8(c2, mG2));
        b8 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, mB0), _mm_shuffle_epi8(c1, mB1)), _mm_shuffle_epi8(c2, mB2));
    }
#endif

    struct Front {
        int W; std::vector<int16_t> prev[3], cur[3];
        explicit Front(int w) : W(w) { for (int p = 0; p < 3; ++p) { prev[p].assign(size_t(w), 0); cur[p].assign(size_t(w), 0); } }

        // Scalar reference band (also used by RLH9_ENC_MODE == 0 and RLH9_NO_SIMD)
        size_t band_scalar(const uint8_t* rgb, int y0, int nb, uint16_t* u) {
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

        // Fused SIMD front (writes U0=zY, U1=zU, US=zU+zV; skips unused U2 plane for 25% less store bandwidth)
        // Zero store-to-load forwarding stalls via register-shift alignr for left neighbor a = cur[x-1].
        size_t band_yus(const uint8_t* rgb, int y0, int nb, uint16_t* U0b, uint16_t* U1b, uint16_t* USb) {
            size_t z = 0;
#if defined(RLH9_SSE) && defined(__AVX2__)
            const __m256i vzero = _mm256_setzero_si256();
#if RLH9_RCT == 1
            const __m256i vk2 = _mm256_set1_epi16(2);
#endif
            auto med_avx = [](__m256i a, __m256i b, __m256i c) __attribute__((always_inline)) {
                const __m256i mn = _mm256_min_epi16(a, b), mx = _mm256_max_epi16(a, b);
                return _mm256_max_epi16(mn, _mm256_min_epi16(mx, _mm256_sub_epi16(_mm256_add_epi16(a, b), c)));
            };
            auto zig_avx = [](__m256i e) __attribute__((always_inline)) {
                return _mm256_xor_si256(_mm256_slli_epi16(e, 1), _mm256_srai_epi16(e, 15));
            };
            auto shift_left1_avx = [](__m256i prev_v, __m256i cur_v) __attribute__((always_inline)) {
                const __m256i hl = _mm256_permute2x128_si256(prev_v, cur_v, 0x21);
                return _mm256_alignr_epi8(cur_v, hl, 14);
            };
#elif defined(RLH9_SSE)
            const __m128i vzero = _mm_setzero_si128();
#if RLH9_RCT == 1
            const __m128i vk2 = _mm_set1_epi16(2);
#endif
            auto med_sse = [](__m128i a, __m128i b, __m128i c) __attribute__((always_inline)) {
                const __m128i mn = _mm_min_epi16(a, b), mx = _mm_max_epi16(a, b);
                return _mm_max_epi16(mn, _mm_min_epi16(mx, _mm_sub_epi16(_mm_add_epi16(a, b), c)));
            };
            auto zig_sse = [](__m128i e) __attribute__((always_inline)) {
                return _mm_xor_si128(_mm_slli_epi16(e, 1), _mm_srai_epi16(e, 15));
            };
            auto shift_left1_sse = [](__m128i prev_v, __m128i cur_v) __attribute__((always_inline)) {
#if defined(__SSSE3__)
                return _mm_alignr_epi8(cur_v, prev_v, 14);
#else
                return _mm_or_si128(_mm_slli_si128(cur_v, 2), _mm_srli_si128(prev_v, 14));
#endif
            };
#endif
            for (int r = 0; r < nb; ++r) {
                const int y = y0 + r; const uint8_t* s = rgb + size_t(y) * W * 3;
                int16_t *C0 = cur[0].data(), *C1 = cur[1].data(), *C2 = cur[2].data();
                const int16_t *Q0 = prev[0].data(), *Q1 = prev[1].data(), *Q2 = prev[2].data();
                uint16_t *U0 = U0b + size_t(r) * W, *U1 = U1b + size_t(r) * W, *US = USb + size_t(r) * W;
                // Prologue x = 0
                {
                    const int R = s[0], G = s[1], B = s[2], U = R - G;
#if RLH9_RCT == 1
                    const int Y = G + ((U + (B - G) + 2) >> 2), V = (B - G) - (U >> 2);
#else
                    const int V = B - G, Y = G + ((U + V) >> 2);
#endif
                    C0[0] = int16_t(Y); C1[0] = int16_t(U); C2[0] = int16_t(V);
                    const uint16_t z0 = zz(y == 0 ? Y : Y - Q0[0]);
                    const uint16_t z1 = zz(y == 0 ? U : U - Q1[0]);
                    const uint16_t z2 = zz(y == 0 ? V : V - Q2[0]);
                    U0[0] = z0; U1[0] = z1; US[0] = uint16_t(z1 + z2);
                    z += size_t(z0 == 0) + size_t(z1 == 0) + size_t(z2 == 0);
                }
                int x = 1;
#if defined(RLH9_SSE) && defined(__AVX2__)
                if (y > 0) {
                    __m256i zacc = vzero; int zblk = 0;
                    __m256i prev_y = _mm256_set1_epi16(C0[0]);
                    __m256i prev_u = _mm256_set1_epi16(C1[0]);
                    __m256i prev_v = _mm256_set1_epi16(C2[0]);
                    for (; x + 16 <= W; x += 16) {
                        __m128i r8, g8, b8;
                        rgb24_deint16_ssse3(s + 3 * x, r8, g8, b8);
                        const __m256i rv = _mm256_cvtepu8_epi16(r8), gv = _mm256_cvtepu8_epi16(g8), bv = _mm256_cvtepu8_epi16(b8);
                        const __m256i uv = _mm256_sub_epi16(rv, gv);
                        const __m256i bg = _mm256_sub_epi16(bv, gv);
#if RLH9_RCT == 1
                        const __m256i yv = _mm256_add_epi16(gv, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(uv, bg), vk2), 2));
                        const __m256i vv = _mm256_sub_epi16(bg, _mm256_srai_epi16(uv, 2));
#else
                        const __m256i vv = bg;
                        const __m256i yv = _mm256_add_epi16(gv, _mm256_srai_epi16(_mm256_add_epi16(uv, vv), 2));
#endif
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C0 + x), yv);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C1 + x), uv);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(C2 + x), vv);
                        const __m256i a0 = shift_left1_avx(prev_y, yv); prev_y = yv;
                        const __m256i a1 = shift_left1_avx(prev_u, uv); prev_u = uv;
                        const __m256i a2 = shift_left1_avx(prev_v, vv); prev_v = vv;
                        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q0 + x));
                        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q1 + x));
                        const __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q2 + x));
                        const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q0 + x - 1));
                        const __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q1 + x - 1));
                        const __m256i c2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Q2 + x - 1));
                        const __m256i u0 = zig_avx(_mm256_sub_epi16(yv, med_avx(a0, b0, c0)));
                        const __m256i u1 = zig_avx(_mm256_sub_epi16(uv, med_avx(a1, b1, c1)));
                        const __m256i u2 = zig_avx(_mm256_sub_epi16(vv, med_avx(a2, b2, c2)));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(U0 + x), u0);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(U1 + x), u1);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(US + x), _mm256_add_epi16(u1, u2));
                        zacc = _mm256_sub_epi16(zacc, _mm256_cmpeq_epi16(u0, vzero));
                        zacc = _mm256_sub_epi16(zacc, _mm256_cmpeq_epi16(u1, vzero));
                        zacc = _mm256_sub_epi16(zacc, _mm256_cmpeq_epi16(u2, vzero));
                        if (++zblk == 8000) {
                            alignas(32) uint16_t za[16]; _mm256_store_si256(reinterpret_cast<__m256i*>(za), zacc);
                            for (int i = 0; i < 16; ++i) z += za[i];
                            zacc = vzero; zblk = 0;
                        }
                    }
                    alignas(32) uint16_t za[16]; _mm256_store_si256(reinterpret_cast<__m256i*>(za), zacc);
                    for (int i = 0; i < 16; ++i) z += za[i];
                }
#elif defined(RLH9_SSE)
                if (y > 0) {
                    __m128i zacc = vzero; int zblk = 0;
                    __m128i prev_y = _mm_set1_epi16(C0[0]);
                    __m128i prev_u = _mm_set1_epi16(C1[0]);
                    __m128i prev_v = _mm_set1_epi16(C2[0]);
                    for (; x + 16 <= W; x += 16) {
                        __m128i r8, g8, b8;
#if defined(__SSSE3__)
                        rgb24_deint16_ssse3(s + 3 * x, r8, g8, b8);
#else
                        alignas(16) uint8_t rb[16], gb[16], bb[16];
                        for (int i = 0; i < 16; ++i) { rb[i] = s[3*(x+i)]; gb[i] = s[3*(x+i)+1]; bb[i] = s[3*(x+i)+2]; }
                        r8 = _mm_load_si128(reinterpret_cast<const __m128i*>(rb));
                        g8 = _mm_load_si128(reinterpret_cast<const __m128i*>(gb));
                        b8 = _mm_load_si128(reinterpret_cast<const __m128i*>(bb));
#endif
                        for (int h = 0; h < 2; ++h) {
                            const int xh = x + h * 8;
                            const __m128i rv = h ? _mm_unpackhi_epi8(r8, vzero) : _mm_unpacklo_epi8(r8, vzero);
                            const __m128i gv = h ? _mm_unpackhi_epi8(g8, vzero) : _mm_unpacklo_epi8(g8, vzero);
                            const __m128i bv = h ? _mm_unpackhi_epi8(b8, vzero) : _mm_unpacklo_epi8(b8, vzero);
                            const __m128i uv = _mm_sub_epi16(rv, gv);
                            const __m128i bg = _mm_sub_epi16(bv, gv);
#if RLH9_RCT == 1
                            const __m128i yv = _mm_add_epi16(gv, _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(uv, bg), vk2), 2));
                            const __m128i vv = _mm_sub_epi16(bg, _mm_srai_epi16(uv, 2));
#else
                            const __m128i vv = bg;
                            const __m128i yv = _mm_add_epi16(gv, _mm_srai_epi16(_mm_add_epi16(uv, vv), 2));
#endif
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(C0 + xh), yv);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(C1 + xh), uv);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(C2 + xh), vv);
                            const __m128i a0 = shift_left1_sse(prev_y, yv); prev_y = yv;
                            const __m128i a1 = shift_left1_sse(prev_u, uv); prev_u = uv;
                            const __m128i a2 = shift_left1_sse(prev_v, vv); prev_v = vv;
                            const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q0 + xh));
                            const __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q1 + xh));
                            const __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q2 + xh));
                            const __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q0 + xh - 1));
                            const __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q1 + xh - 1));
                            const __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Q2 + xh - 1));
                            const __m128i u0 = zig_sse(_mm_sub_epi16(yv, med_sse(a0, b0, c0)));
                            const __m128i u1 = zig_sse(_mm_sub_epi16(uv, med_sse(a1, b1, c1)));
                            const __m128i u2 = zig_sse(_mm_sub_epi16(vv, med_sse(a2, b2, c2)));
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(U0 + xh), u0);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(U1 + xh), u1);
                            _mm_storeu_si128(reinterpret_cast<__m128i*>(US + xh), _mm_add_epi16(u1, u2));
                            zacc = _mm_sub_epi16(zacc, _mm_cmpeq_epi16(u0, vzero));
                            zacc = _mm_sub_epi16(zacc, _mm_cmpeq_epi16(u1, vzero));
                            zacc = _mm_sub_epi16(zacc, _mm_cmpeq_epi16(u2, vzero));
                        }
                        if (++zblk == 4000) {
                            alignas(16) uint16_t za[8]; _mm_store_si128(reinterpret_cast<__m128i*>(za), zacc);
                            for (int i = 0; i < 8; ++i) z += za[i];
                            zacc = vzero; zblk = 0;
                        }
                    }
                    alignas(16) uint16_t za[8]; _mm_store_si128(reinterpret_cast<__m128i*>(za), zacc);
                    for (int i = 0; i < 8; ++i) z += za[i];
                }
#endif
                // Scalar tail (also handles y == 0)
                int a0 = C0[x - 1], a1 = C1[x - 1], a2 = C2[x - 1];
                if (y == 0) {
                    for (; x < W; ++x) {
                        const int R = s[3 * x], G = s[3 * x + 1], B = s[3 * x + 2], U = R - G;
#if RLH9_RCT == 1
                        const int Y = G + ((U + (B - G) + 2) >> 2), V = (B - G) - (U >> 2);
#else
                        const int V = B - G, Y = G + ((U + V) >> 2);
#endif
                        C0[x] = int16_t(Y); C1[x] = int16_t(U); C2[x] = int16_t(V);
                        const uint16_t z0 = zz(Y - a0), z1 = zz(U - a1), z2 = zz(V - a2);
                        U0[x] = z0; U1[x] = z1; US[x] = uint16_t(z1 + z2);
                        z += size_t(z0 == 0) + size_t(z1 == 0) + size_t(z2 == 0);
                        a0 = Y; a1 = U; a2 = V;
                    }
                } else {
                    for (; x < W; ++x) {
                        const int R = s[3 * x], G = s[3 * x + 1], B = s[3 * x + 2], U = R - G;
#if RLH9_RCT == 1
                        const int Y = G + ((U + (B - G) + 2) >> 2), V = (B - G) - (U >> 2);
#else
                        const int V = B - G, Y = G + ((U + V) >> 2);
#endif
                        C0[x] = int16_t(Y); C1[x] = int16_t(U); C2[x] = int16_t(V);
                        const uint16_t z0 = zz(Y - med(a0, Q0[x], Q0[x - 1]));
                        const uint16_t z1 = zz(U - med(a1, Q1[x], Q1[x - 1]));
                        const uint16_t z2 = zz(V - med(a2, Q2[x], Q2[x - 1]));
                        U0[x] = z0; U1[x] = z1; US[x] = uint16_t(z1 + z2);
                        z += size_t(z0 == 0) + size_t(z1 == 0) + size_t(z2 == 0);
                        a0 = Y; a1 = U; a2 = V;
                    }
                }
                for (int p = 0; p < 3; ++p) std::swap(prev[p], cur[p]);
            }
            return z;
        }

        size_t band(const uint8_t* rgb, int y0, int nb, uint16_t* u) {
            return band_scalar(rgb, y0, nb, u);
        }
    };

    // ---------------- token walk of one lane row (training and reference encoder share it) ----------------
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

    struct BitW {                                         // original branchy BitW
        uint8_t* ptr; uint64_t acc = 0; int n = 0;
        inline void put(uint64_t v, int k) { acc = (acc << k) | v; n += k; if (n >= 32) { n -= 32; const uint32_t w = __builtin_bswap32(uint32_t(acc >> n)); std::memcpy(ptr, &w, 4); ptr += 4; } }
        inline void put_tbin(uint32_t x, uint32_t m) { const int k = bw(m) - 1; const uint32_t t = (2u << k) - m; if (x < t) put(x, k); else put(x + t, k + 1); }
        inline uint8_t* flush() { while (n >= 8) { n -= 8; *ptr++ = uint8_t(acc >> n); } if (n > 0) { *ptr++ = uint8_t(acc << (8 - n)); n = 0; } acc = 0; return ptr; }
    };

    struct BitW_Fast {                                    // branchless 64-bit writer: invariant n in 0..7, accepts k in 1..56
        uint8_t* ptr; uint64_t acc = 0; int n = 0;
        inline void put(uint64_t v, int k) {
            acc = (acc << k) | v;
            n += k;
            const uint64_t w = __builtin_bswap64(acc << (64 - n));
            std::memcpy(ptr, &w, 8);
            ptr += (n >> 3);
            n &= 7;
        }
        inline uint8_t* flush() {
            if (n > 0) { *ptr++ = uint8_t(acc << (8 - n)); n = 0; }
            acc = 0; return ptr;
        }
    };

    // Helpers for LUT vs ALU encoding of value class, run class, and zU|S
    inline void enc_val_info(const Tables& T, uint32_t v, uint32_t& cv, uint32_t& vm, uint32_t& vh, uint32_t& vman) {
#if RLH9_VENC_MODE == 0
        const auto& vi = T.venc[v]; cv = vi.cv; vm = vi.vm; vh = vi.vh; vman = vi.vman;
#elif RLH9_VENC_MODE == 1
        const uint32_t p = T.venc32[v];
        cv = p & 63u; vm = (p >> 6) & 15u; vh = (p >> 10) & 15u; vman = p >> 14;
#else
        // Pure ALU: v >= 1
        const uint32_t b = 32u - uint32_t(__builtin_clz(v));
        if (b <= 1u) { cv = 1u; vm = 0u; vh = 0u; vman = 0u; }
        else if (b == 2u) { cv = 4u + (v & 1u); vm = 0u; vh = 1u + (v & 1u); vman = 0u; }
        else {
            const uint32_t sh = b - 3u;
            const uint32_t q = (v >> sh) & 3u;
            cv = 4u * b - 4u + q;
            vm = sh;
            vman = v & ((1u << sh) - 1u);
            vh = (b >= 9u) ? 15u : (2u * b - 3u + (q >> 1));
        }
#endif
    }

    inline void enc_run_info(const Tables& T, uint32_t rl, uint32_t& rs, uint32_t& rm, uint32_t& rc, uint32_t& rman) {
#if RLH9_RENC_MODE == 0
        if (__builtin_expect(rl < 256u, 1)) {
            const auto& ri = T.renc[rl]; rs = ri.rs; rm = ri.rm; rc = ri.rc; rman = ri.rman;
        } else {
            const uint32_t b = 32u - uint32_t(__builtin_clz(rl));
            rs = 4u + b; rm = b - 1u; rc = b > 7u ? 7u : b; rman = rl - (1u << rm);
        }
#elif RLH9_RENC_MODE == 1
        if (__builtin_expect(rl < 256u, 1)) {
            const uint32_t p = T.renc32[rl];
            rs = p & 31u; rm = (p >> 5) & 15u; rc = (p >> 9) & 7u; rman = p >> 12;
        } else {
            const uint32_t b = 32u - uint32_t(__builtin_clz(rl));
            rs = 4u + b; rm = b - 1u; rc = b > 7u ? 7u : b; rman = rl - (1u << rm);
        }
#else
        if (__builtin_expect(rl < 8u, 1)) {
            rs = rl; rm = 0u; rman = 0u;
            rc = rl ? (32u - uint32_t(__builtin_clz(rl))) : 0u;
        } else {
            const uint32_t b = 32u - uint32_t(__builtin_clz(rl));
            rs = 4u + b; rm = b - 1u; rc = b > 7u ? 7u : b; rman = rl - (1u << rm);
        }
#endif
    }

    inline void enc_zu_code(const Tables& T, uint32_t sc, uint32_t sv, uint32_t zu, uint32_t& rcode, uint32_t& rlen) {
#if RLH9_ENC_MODE == 6 || RLH9_ENC_MODE == 8 || RLH9_TBIN_MODE == 2
        // Fully branchless cmov: RemLUT for sv <= 15, tbin_tbl[sv] for sv >= 16
        const uint32_t rc = T.rem_cw[sc][sv & 15u][zu & 15u];
        const uint32_t tb = T.tbin_tbl[sv];
        const uint32_t tk = tb & 15u, tt = tb >> 4;
        const uint32_t ge = uint32_t(zu >= tt);
        const uint32_t tcode = zu + (ge ? tt : 0u);
        const uint32_t tlen  = tk + ge;
        const bool is_rem = (sv <= uint32_t(REMS));
        rcode = is_rem ? (rc & 255u) : tcode;
        rlen  = is_rem ? (rc >> 8)   : tlen;
#elif RLH9_TBIN_MODE == 1
        if (__builtin_expect(sv <= uint32_t(REMS), 1)) {
            const uint32_t rc = T.rem_cw[sc][sv][zu];
            rcode = rc & 255u; rlen = rc >> 8;
        } else {
            const uint32_t tb = T.tbin_tbl[sv];
            const uint32_t tk = tb & 15u, tt = tb >> 4;
            const uint32_t ge = uint32_t(zu >= tt);
            rcode = zu + (ge ? tt : 0u);
            rlen  = tk + ge;
        }
#else
        if (__builtin_expect(sv <= uint32_t(REMS), 1)) {
            const uint32_t rc = T.rem_cw[sc][sv][zu];
            rcode = rc & 255u; rlen = rc >> 8;
        } else {
            const uint32_t m = sv + 1u;
            const uint32_t tk = 31u - uint32_t(__builtin_clz(m));
            const uint32_t tt = (2u << tk) - m;
            const uint32_t ge = uint32_t(zu >= tt);
            rcode = zu + (ge ? tt : 0u);
            rlen  = tk + ge;
        }
#endif
    }

    // Emit one non-zero Y token into BitW_Fast (single put, max 36 bits)
    inline void emit_token_y(const Tables& T, const Fsm& M, BitW_Fast& w, uint32_t gb, uint32_t cm, uint32_t cc,
                             uint32_t& st, uint32_t rl, uint32_t v) {
        uint32_t rs, rm, rc, rman, cv, vm, vh, vman;
        enc_run_info(T, rl, rs, rm, rc, rman);
        enc_val_info(T, v, cv, vm, vh, vman);
        const uint32_t tc = gb + ctx_of(st, cm, cc);
        const uint32_t cw = T.enc_cw[tc][rs * NV + cv];
        const int nm = int(rm + vm);
        const uint64_t bits = (uint64_t(cw & 0xFFFFu) << nm) | (uint64_t(rman) << vm) | vman;
        w.put(bits, int(cw >> 16) + nm);
        st = next_st(M, st, int(vh), int(rc));
    }

    // Emit one non-zero S token + zU|S into BitW_Fast (single put, max 47 bits)
    inline void emit_token_s(const Tables& T, const Fsm& M, BitW_Fast& w, uint32_t gb, uint32_t sc, uint32_t cm, uint32_t cc,
                             uint32_t& st, uint32_t rl, uint32_t sv, uint32_t zu) {
        uint32_t rs, rm, rc, rman, cv, vm, vh, vman;
        enc_run_info(T, rl, rs, rm, rc, rman);
        enc_val_info(T, sv, cv, vm, vh, vman);
        const uint32_t tc = gb + ctx_of(st, cm, cc);
        const uint32_t cw = T.enc_cw[tc][rs * NV + cv];
        const int nm = int(rm + vm);
        uint32_t rcode, rlen;
        enc_zu_code(T, sc, sv, zu, rcode, rlen);
        const uint64_t man = (uint64_t(rman) << vm) | vman;
        const uint64_t bits = (((uint64_t(cw & 0xFFFFu) << nm) | man) << rlen) | rcode;
        w.put(bits, int(cw >> 16) + nm + int(rlen));
        st = next_st(M, st, int(vh), int(rc));
    }

    inline void emit_eob(const Tables& T, BitW_Fast& w, uint32_t gb, uint32_t cm, uint32_t cc, uint32_t st) {
        const uint32_t cw = T.enc_cw[gb + ctx_of(st, cm, cc)][EOB];
        w.put(cw & 0xFFFFu, int(cw >> 16));
    }

    // Non-zero 16-bit bitmask of 16 uint16_t elements (bit i == 1 iff row[i] != 0)
    inline uint32_t nz_mask16(const uint16_t* p) {
#if defined(RLH9_SSE) && defined(__AVX2__)
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i eq = _mm256_cmpeq_epi16(v, _mm256_setzero_si256());
        const uint32_t m32 = uint32_t(_mm256_movemask_epi8(eq));
        // Compact even bits of ~m32: pack 16-bit eq into 8-bit via packs_epi16
        (void)m32;
        const __m128i lo = _mm256_castsi256_si128(eq);
        const __m128i hi = _mm256_extracti128_si256(eq, 1);
        const __m128i pk = _mm_packs_epi16(lo, hi);
        return uint32_t(~_mm_movemask_epi8(pk)) & 0xFFFFu;
#elif defined(RLH9_SSE)
        const __m128i z = _mm_setzero_si128();
        const __m128i e0 = _mm_cmpeq_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)), z);
        const __m128i e1 = _mm_cmpeq_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8)), z);
        const __m128i pk = _mm_packs_epi16(e0, e1);
        return uint32_t(~_mm_movemask_epi8(pk)) & 0xFFFFu;
#else
        uint32_t m = 0;
        for (int i = 0; i < 16; ++i) m |= uint32_t(p[i] != 0) << i;
        return m;
#endif
    }

    // Single-lane row encoders: bitmask zero-skip vs simple scan
    inline void encode_row_y_mask(const Tables& T, const Fsm& M, const uint16_t* row, int W,
                                  uint32_t gb, uint32_t cm, uint32_t cc, uint32_t& st, BitW_Fast& w) {
        int x = 0; uint32_t rl = 0;
        for (; x + 16 <= W; x += 16) {
            uint32_t m = nz_mask16(row + x);
            if (!m) { rl += 16; continue; }
            int prev = 0;
            while (m) {
                const int tz = __builtin_ctz(m);
                rl += uint32_t(tz - prev);
                emit_token_y(T, M, w, gb, cm, cc, st, rl, row[x + tz]);
                rl = 0; prev = tz + 1;
                m &= m - 1u;
            }
            rl += uint32_t(16 - prev);
        }
        for (; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            emit_token_y(T, M, w, gb, cm, cc, st, rl, v);
            rl = 0;
        }
        if (rl) emit_eob(T, w, gb, cm, cc, st);
    }

    inline void encode_row_s_mask(const Tables& T, const Fsm& M, const uint16_t* row, const uint16_t* rowu, int W,
                                  uint32_t gb, uint32_t sc, uint32_t cm, uint32_t cc, uint32_t& st, BitW_Fast& w) {
        int x = 0; uint32_t rl = 0;
        for (; x + 16 <= W; x += 16) {
            uint32_t m = nz_mask16(row + x);
            if (!m) { rl += 16; continue; }
            int prev = 0;
            while (m) {
                const int tz = __builtin_ctz(m);
                rl += uint32_t(tz - prev);
                const int idx = x + tz;
                emit_token_s(T, M, w, gb, sc, cm, cc, st, rl, row[idx], rowu[idx]);
                rl = 0; prev = tz + 1;
                m &= m - 1u;
            }
            rl += uint32_t(16 - prev);
        }
        for (; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            emit_token_s(T, M, w, gb, sc, cm, cc, st, rl, v, rowu[x]);
            rl = 0;
        }
        if (rl) emit_eob(T, w, gb, cm, cc, st);
    }

    inline void encode_row_y_simple(const Tables& T, const Fsm& M, const uint16_t* row, int W,
                                    uint32_t gb, uint32_t cm, uint32_t cc, uint32_t& st, BitW_Fast& w) {
        uint32_t rl = 0;
        for (int x = 0; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            emit_token_y(T, M, w, gb, cm, cc, st, rl, v);
            rl = 0;
        }
        if (rl) emit_eob(T, w, gb, cm, cc, st);
    }

    inline void encode_row_s_simple(const Tables& T, const Fsm& M, const uint16_t* row, const uint16_t* rowu, int W,
                                    uint32_t gb, uint32_t sc, uint32_t cm, uint32_t cc, uint32_t& st, BitW_Fast& w) {
        uint32_t rl = 0;
        for (int x = 0; x < W; ++x) {
            const uint32_t v = row[x];
            if (!v) { ++rl; continue; }
            emit_token_s(T, M, w, gb, sc, cm, cc, st, rl, v, rowu[x]);
            rl = 0;
        }
        if (rl) emit_eob(T, w, gb, cm, cc, st);
    }

    // 2-lane interleaved stepper state (fits cleanly in GPRs for 2 concurrent lanes)
    struct LaneCursor {
        const uint16_t* plane;
        const uint16_t* plane_u;
        const uint16_t* row;
        const uint16_t* rowu;
        int r, r_end, x, W;
        uint32_t st;
        BitW_Fast w;

        void init(const uint16_t* P, const uint16_t* PU, int r0, int r1, int width, uint8_t* dst) {
            plane = P; plane_u = PU; r = r0; r_end = r1; x = 0; W = width;
            row = P + size_t(r0) * width;
            rowu = PU ? (PU + size_t(r0) * width) : nullptr;
            st = ST0; w = BitW_Fast{dst};
        }
        inline void next_row() {
            ++r; x = 0;
            row += W;
            if (rowu) rowu += W;
        }
    };

    inline void step_lane_y(const Tables& T, const Fsm& M, LaneCursor& L, uint32_t gb, uint32_t cm, uint32_t cc) {
        const uint16_t* row = L.row;
        const int W = L.W;
        int x = L.x;
        uint32_t rl = 0;
        while (x < W && row[x] == 0) { ++x; ++rl; }
        if (x < W) {
            emit_token_y(T, M, L.w, gb, cm, cc, L.st, rl, row[x]);
            L.x = x + 1;
            if (L.x == W) L.next_row();
        } else {
            emit_eob(T, L.w, gb, cm, cc, L.st);
            L.next_row();
        }
    }

    inline void step_lane_s(const Tables& T, const Fsm& M, LaneCursor& L, uint32_t gb, uint32_t sc, uint32_t cm, uint32_t cc) {
        const uint16_t* row = L.row;
        const int W = L.W;
        int x = L.x;
        uint32_t rl = 0;
        while (x < W && row[x] == 0) { ++x; ++rl; }
        if (x < W) {
            emit_token_s(T, M, L.w, gb, sc, cm, cc, L.st, rl, row[x], L.rowu[x]);
            L.x = x + 1;
            if (L.x == W) L.next_row();
        } else {
            emit_eob(T, L.w, gb, cm, cc, L.st);
            L.next_row();
        }
    }

    // lane l of a band of nb rows: plane (0 = Y, 1 = S) and rows [r0, r1)
    inline void lane_rows(int l, int nb, int& plane, int& r0, int& r1) { const int m = nb / 2; plane = l >> 1; r0 = (l & 1) ? m : 0; r1 = (l & 1) ? nb : m; }

    // encode the R8F4 body (no file header) of a W x H 8-bit RGB image
    inline std::vector<uint8_t> encode(const uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: width must be 1..16383");
        const Tables& T = tables();
        std::vector<uint8_t> out; out.reserve(size_t(W) * H + 4096);
        Front fe(W);
        const size_t LB = size_t(HB / 2) * W * 8 + 64;   // <= 47 bits per pixel token + 8B BitW_Fast slack
        std::vector<uint8_t> lb[NLANE]; for (auto& b : lb) b.resize(LB);
#if RLH9_ENC_MODE == 0
        std::vector<uint16_t> bu(size_t(4) * HB * W);
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
#else
        const Fsm& M = fsm();
        std::vector<uint16_t> bu(size_t(3) * HB * W);    // U0 (Y), U1 (zU), US (zU+zV)
        for (int y0 = 0; y0 < H; y0 += HB) {
            const int nb = std::min(HB, H - y0); const size_t PS = size_t(nb) * W;
            uint16_t* U0 = bu.data(); uint16_t* U1 = U0 + PS; uint16_t* US = U1 + PS;
            const size_t z = fe.band_yus(rgb, y0, nb, U0, U1, US);
            const uint32_t sc = uint32_t(z * 100 >= size_t(RLH9_ZTHR) * 3 * PS);
            const uint32_t cm = ctx_m(sc), cc = ctx_c(sc);
            const uint32_t gy = sc * SETC, gs = gy + NCTX;
            const int m = nb / 2;
            uint32_t ls[NLANE];
#if RLH9_ENC_MODE == 2
            // Idea 1a: 2-lane interleaved homogeneous pairs (Y0+Y1, then S0+S1)
            if (sc) {
                for (int l = 0; l < 2; ++l) {
                    const int r0 = l ? m : 0, r1 = l ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_y_mask(T, M, U0 + size_t(r) * W, W, gy, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
                for (int l = 2; l < 4; ++l) {
                    const int r0 = (l & 1) ? m : 0, r1 = (l & 1) ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_s_mask(T, M, US + size_t(r) * W, U1 + size_t(r) * W, W, gs, sc, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
            } else {
                LaneCursor L0, L1;
                L0.init(U0, nullptr, 0, m, W, lb[0].data());
                L1.init(U0, nullptr, m, nb, W, lb[1].data());
                while (L0.r < L0.r_end && L1.r < L1.r_end) { step_lane_y(T, M, L0, gy, cm, cc); step_lane_y(T, M, L1, gy, cm, cc); }
                while (L0.r < L0.r_end) step_lane_y(T, M, L0, gy, cm, cc);
                while (L1.r < L1.r_end) step_lane_y(T, M, L1, gy, cm, cc);
                ls[0] = uint32_t(L0.w.flush() - lb[0].data());
                ls[1] = uint32_t(L1.w.flush() - lb[1].data());

                LaneCursor L2, L3;
                L2.init(US, U1, 0, m, W, lb[2].data());
                L3.init(US, U1, m, nb, W, lb[3].data());
                while (L2.r < L2.r_end && L3.r < L3.r_end) { step_lane_s(T, M, L2, gs, sc, cm, cc); step_lane_s(T, M, L3, gs, sc, cm, cc); }
                while (L2.r < L2.r_end) step_lane_s(T, M, L2, gs, sc, cm, cc);
                while (L3.r < L3.r_end) step_lane_s(T, M, L3, gs, sc, cm, cc);
                ls[2] = uint32_t(L2.w.flush() - lb[2].data());
                ls[3] = uint32_t(L3.w.flush() - lb[3].data());
            }
#elif RLH9_ENC_MODE == 3 || RLH9_ENC_MODE == 7
            // Idea 1b: 2-lane interleaved mixed pairs (Y0+S0, then Y1+S1)
            if (sc) {
                for (int l = 0; l < 2; ++l) {
                    const int r0 = l ? m : 0, r1 = l ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_y_mask(T, M, U0 + size_t(r) * W, W, gy, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
                for (int l = 2; l < 4; ++l) {
                    const int r0 = (l & 1) ? m : 0, r1 = (l & 1) ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_s_mask(T, M, US + size_t(r) * W, U1 + size_t(r) * W, W, gs, sc, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
            } else {
                LaneCursor LY0, LS0;
                LY0.init(U0, nullptr, 0, m, W, lb[0].data());
                LS0.init(US, U1, 0, m, W, lb[2].data());
                while (LY0.r < LY0.r_end && LS0.r < LS0.r_end) { step_lane_y(T, M, LY0, gy, cm, cc); step_lane_s(T, M, LS0, gs, sc, cm, cc); }
                while (LY0.r < LY0.r_end) step_lane_y(T, M, LY0, gy, cm, cc);
                while (LS0.r < LS0.r_end) step_lane_s(T, M, LS0, gs, sc, cm, cc);
                ls[0] = uint32_t(LY0.w.flush() - lb[0].data());
                ls[2] = uint32_t(LS0.w.flush() - lb[2].data());

                LaneCursor LY1, LS1;
                LY1.init(U0, nullptr, m, nb, W, lb[1].data());
                LS1.init(US, U1, m, nb, W, lb[3].data());
                while (LY1.r < LY1.r_end && LS1.r < LS1.r_end) { step_lane_y(T, M, LY1, gy, cm, cc); step_lane_s(T, M, LS1, gs, sc, cm, cc); }
                while (LY1.r < LY1.r_end) step_lane_y(T, M, LY1, gy, cm, cc);
                while (LS1.r < LS1.r_end) step_lane_s(T, M, LS1, gs, sc, cm, cc);
                ls[1] = uint32_t(LY1.w.flush() - lb[1].data());
                ls[3] = uint32_t(LS1.w.flush() - lb[3].data());
            }
#else
#if RLH9_ENC_MODE == 4 || RLH9_ENC_MODE == 8
            const bool use_mask = (sc != 0);
#elif RLH9_ENC_MODE == 5
            const bool use_mask = (z * 100 >= size_t(RLH9_MASK_THR) * 3 * PS);
#else
            const bool use_mask = true;
#endif
            if (use_mask) {
                for (int l = 0; l < 2; ++l) {
                    const int r0 = l ? m : 0, r1 = l ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_y_mask(T, M, U0 + size_t(r) * W, W, gy, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
                for (int l = 2; l < 4; ++l) {
                    const int r0 = (l & 1) ? m : 0, r1 = (l & 1) ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_s_mask(T, M, US + size_t(r) * W, U1 + size_t(r) * W, W, gs, sc, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
            } else {
                for (int l = 0; l < 2; ++l) {
                    const int r0 = l ? m : 0, r1 = l ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_y_simple(T, M, U0 + size_t(r) * W, W, gy, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
                for (int l = 2; l < 4; ++l) {
                    const int r0 = (l & 1) ? m : 0, r1 = (l & 1) ? nb : m;
                    BitW_Fast w{lb[l].data()}; uint32_t st = ST0;
                    for (int r = r0; r < r1; ++r) encode_row_s_simple(T, M, US + size_t(r) * W, U1 + size_t(r) * W, W, gs, sc, cm, cc, st, w);
                    ls[l] = uint32_t(w.flush() - lb[l].data());
                }
            }
#endif
            if (ls[0] >= 0x80000000u) throw std::runtime_error("rlh9: band too large");
            const size_t o = out.size(); out.resize(o + 4 * NLANE + ls[0] + ls[1] + ls[2] + ls[3]);
            uint8_t* d = out.data() + o; const uint32_t hd[NLANE] = {ls[0] | sc << 31, ls[1], ls[2], ls[3]}; std::memcpy(d, hd, 4 * NLANE); d += 4 * NLANE;
            for (int l = 0; l < NLANE; ++l) { std::memcpy(d, lb[l].data(), ls[l]); d += ls[l]; }
        }
#endif
        return out;
    }

    // ---------------- decoder ----------------
    inline uint64_t peek(const uint8_t* base, uint64_t pos) { uint64_t w; std::memcpy(&w, base + (pos >> 3), 8); return __builtin_bswap64(w) << (pos & 7); }

    __attribute__((noinline, cold)) inline SlowSym decode_slow_loop(const Tables& T, uint32_t tc, uint64_t w, bool& bad) {
        for (int l = 1; l <= MAXL; ++l) {
            const uint32_t v = uint32_t(w >> (64 - l)), f = T.first[tc][l];
            if (v - f < T.cntl[tc][l]) { SlowSym r = T.si[T.sorted[tc][T.offs[tc][l] + v - f]]; r.len = uint8_t(l); return r; }
        }
        bad = true; SlowSym r = T.si[EOB]; r.len = 1; return r;
    }

    inline SlowSym decode_slow_simd(const Tables& T, uint32_t tc, uint64_t w, bool& bad) {
        const uint32_t v16 = uint32_t(w >> 48);
        const int16_t vb = int16_t(v16 ^ 0x8000u);
        uint32_t gt_mask = T.lim_true_mask[tc];
#if defined(RLH9_SSE) && defined(__AVX2__)
        const __m256i vvec = _mm256_set1_epi16(vb);
        const __m256i lim = _mm256_load_si256(reinterpret_cast<const __m256i*>(T.lim16[tc]));
        const uint32_t m32 = uint32_t(_mm256_movemask_epi8(_mm256_cmpgt_epi16(lim, vvec)));
        // Every 16-bit lane produces 2 identical bits in m32 -> extract even bits via PEXT or pack
        const __m128i lo = _mm256_castsi256_si128(_mm256_cmpgt_epi16(lim, vvec));
        const __m128i hi = _mm256_extracti128_si256(_mm256_cmpgt_epi16(lim, vvec), 1);
        (void)m32;
        gt_mask |= uint32_t(_mm_movemask_epi8(_mm_packs_epi16(lo, hi)));
#elif defined(RLH9_SSE)
        const __m128i vvec = _mm_set1_epi16(vb);
        const __m128i c0 = _mm_cmpgt_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(T.lim16[tc])), vvec);
        const __m128i c1 = _mm_cmpgt_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(T.lim16[tc] + 8)), vvec);
        gt_mask |= uint32_t(_mm_movemask_epi8(_mm_packs_epi16(c0, c1)));
#else
        for (int i = 0; i < 14; ++i) gt_mask |= uint32_t(T.lim16[tc][i] > vb) << i;
#endif
        const int l = 15 - __builtin_popcount(gt_mask & 0x3FFFu);
        const uint32_t v = uint32_t(w >> (64 - l)), f = T.first[tc][l];
        if (__builtin_expect(v - f < T.cntl[tc][l], 1)) {
            SlowSym r = T.si[T.sorted[tc][T.offs[tc][l] + v - f]];
            r.len = uint8_t(l);
            return r;
        }
        bad = true; SlowSym r = T.si[EOB]; r.len = 1; return r;
    }

    __attribute__((noinline)) inline bool entropy_band(const Tables& T, const uint8_t* base, uint16_t* const u,
                                                       const uint64_t pos[NLANE], uint32_t W, uint32_t nb, uint32_t sc) {
        const uint32_t PS = W * nb, MW = (nb / 2) * W;
        uint64_t p0 = pos[0], p1 = pos[1], p2 = pos[2], p3 = pos[3];
        uint32_t s0 = ST0, s1 = ST0, s2 = ST0, s3 = ST0;
        uint32_t q0 = 0, qe0 = W, q1 = MW, qe1 = MW + W, q2 = PS, qe2 = PS + W, q3 = PS + MW, qe3 = PS + MW + W;
        const uint32_t e0 = MW, e1 = PS, e2 = PS + MW, e3 = 2 * PS, gy = sc * SETC, gs = gy + NCTX, cm = ctx_m(sc), cc = ctx_c(sc);
#if RLH9_DEC_LUT64 == 1
        const uint64_t* const lut64 = &T.lut64[0][0];
#else
        const uint32_t* const lut = &T.lut[0][0];
        const uint8_t* const lcl = &T.lut_cls[0][0];
#endif
#if RLH9_DEC_SLOW_MODE == 1
        const uint16_t* const sub_lut = T.sub_lut.data();
#endif
        const uint16_t* const rlut = &T.rem_lut[sc][0][0];
#if RLH9_DEC_TBIN_MODE >= 1
        const uint16_t* const tbtbl = T.tbin_tbl;
#endif
        bool bad = false;
        const Fsm& M = fsm();

#if RLH9_DEC_NO_MEMSET == 1
#define RLH9_ZERO_Y(Q, CNT) do { const uint32_t zc_ = (CNT); if (zc_) std::memset(u + (Q), 0, size_t(zc_) * sizeof(uint16_t)); } while (0)
#define RLH9_ZERO_S(Q, CNT) do { const uint32_t zc_ = (CNT); if (zc_) { std::memset(u + (Q), 0, size_t(zc_) * sizeof(uint16_t)); std::memset(u + (Q) + PS, 0, size_t(zc_) * sizeof(uint16_t)); } } while (0)
#else
#define RLH9_ZERO_Y(Q, CNT) do {} while (0)
#define RLH9_ZERO_S(Q, CNT) do {} while (0)
#endif
#if RLH9_DEC_NO_MEMSET == 1
#define RLH9_PUT_Y(Q, QE, RL, V) if (__builtin_expect((RL) >= (QE) - (Q), 0)) { RLH9_ZERO_Y(Q, (QE) - (Q)); (Q) = (QE); bad = true; } else { RLH9_ZERO_Y(Q, RL); (Q) += (RL); u[(Q)++] = uint16_t(V); }
#define RLH9_PUT_S(Q, QE, RL, W_, SV_, U_, P_) if (__builtin_expect((RL) >= (QE) - (Q), 0)) { RLH9_ZERO_S(Q, (QE) - (Q)); (Q) = (QE); bad = true; } else { RLH9_ZERO_S(Q, RL); (Q) += (RL); RLH9_TBIN(W_, SV_, U_, P_, Q) }
#else
#define RLH9_PUT_Y(Q, QE, RL, V) (Q) += (RL); u[(Q)++] = uint16_t(V);
#define RLH9_PUT_S(Q, QE, RL, W_, SV_, U_, P_) (Q) += (RL); RLH9_TBIN(W_, SV_, U_, P_, Q)
#endif

#if RLH9_DEC_LUT64 == 1
#define RLH9_HEAD(POS, ST, GB) \
        const uint64_t w_ = peek(base, POS); \
        const uint32_t tc_ = (GB) + ctx_of(ST, cm, cc); \
        const size_t li_ = (size_t(tc_) << LUTB) | uint32_t(w_ >> (64 - LUTB)); \
        const uint64_t e64_ = lut64[li_]; \
        const uint32_t e_ = uint32_t(e64_);
#else
#define RLH9_HEAD(POS, ST, GB) \
        const uint64_t w_ = peek(base, POS); \
        const uint32_t tc_ = (GB) + ctx_of(ST, cm, cc); \
        const size_t li_ = (size_t(tc_) << LUTB) | uint32_t(w_ >> (64 - LUTB)); \
        const uint32_t e_ = lut[li_]; const uint32_t cl_ = lcl[li_];
#endif

#if RLH9_DEC_SLOW_MODE == 1
#define RLH9_GET_SLOW(SS) \
        const uint16_t se_ = sub_lut[(size_t(e_ >> 4) << (MAXL - LUTB)) | uint32_t((w_ >> (64 - MAXL)) & ((1u << (MAXL - LUTB)) - 1u))]; \
        SlowSym SS = T.si[se_ & 1023u]; SS.len = uint8_t(se_ >> 10);
#elif RLH9_DEC_SLOW_MODE == 2
#define RLH9_GET_SLOW(SS) const SlowSym SS = decode_slow_simd(T, tc_, w_, bad);
#else
#define RLH9_GET_SLOW(SS) const SlowSym SS = decode_slow_loop(T, tc_, w_, bad);
#endif

#define RLH9_TAIL(Q, QE) if (__builtin_expect(Q >= QE, 0)) { bad |= Q > QE; Q = QE; QE += W; }

#if RLH9_DEC_LUT64 == 1
#define RLH9_FAST_DECODE(MVAL, RLVAL, VVAL, ST) \
        const uint32_t nm_ = (e_ >> 4) & 31u, vm_ = (e_ >> 9) & 7u; \
        const uint32_t sh_ = uint32_t((e64_ >> 48) & 63u); \
        const uint32_t MVAL = uint32_t(((w_ << len_) >> 1) >> sh_); \
        const uint32_t RLVAL = ((e_ >> 23) & 63u) + (MVAL >> vm_); \
        const uint32_t VVAL = ((e_ >> 12) & 2047u) + (MVAL & uint32_t((e64_ >> 40) & 255u)); \
        const uint32_t vh_ = uint32_t((e64_ >> 32) & 15u), rc_ = uint32_t((e64_ >> 36) & 7u);
#define RLH9_UPDATE_ST_FAST(ST) ST = uint32_t(M.v[ST & 63u][vh_]) | uint32_t(M.r6[ST >> 6][rc_]);
#else
#define RLH9_FAST_DECODE(MVAL, RLVAL, VVAL, ST) \
        const uint32_t nm_ = (e_ >> 4) & 31u, vm_ = (e_ >> 9) & 7u; \
        const uint32_t MVAL = uint32_t(((w_ << len_) >> 1) >> (63 - nm_)); \
        const uint32_t RLVAL = ((e_ >> 23) & 63u) + (MVAL >> vm_); \
        const uint32_t VVAL = ((e_ >> 12) & 2047u) + (MVAL & ((1u << vm_) - 1u));
#define RLH9_UPDATE_ST_FAST(ST) ST = uint32_t(M.v[ST & 63u][cl_ & 15u]) | uint32_t(M.r6[ST >> 6][cl_ >> 4]);
#endif

#define RLH9_STEP_Y(POS, ST, Q, QE) do { RLH9_HEAD(POS, ST, gy) \
        if (__builtin_expect((e_ & 15u) != 0, 1)) { \
            const uint32_t len_ = e_ & 15u; \
            if (__builtin_expect(int32_t(e_) < 0, 0)) { POS += len_; RLH9_ZERO_Y(Q, QE - Q); Q = QE; } \
            else { \
                RLH9_FAST_DECODE(m_, rl_, v_, ST) \
                POS += len_ + nm_; \
                RLH9_PUT_Y(Q, QE, rl_, v_) \
                RLH9_UPDATE_ST_FAST(ST) \
            } \
        } else { \
            RLH9_GET_SLOW(ss_) \
            const uint32_t m_ = uint32_t(((w_ << ss_.len) >> 1) >> (63 - ss_.nm)); \
            POS += ss_.len + ss_.nm; \
            if (ss_.eob) { RLH9_ZERO_Y(Q, QE - Q); Q = QE; } \
            else { \
                uint32_t rl_ = ss_.rbase + (m_ >> ss_.vm); const uint32_t v_ = ss_.vbase + (m_ & ss_.vmask); \
                if (rl_ >= QE - Q) { rl_ = QE - Q - 1; bad = true; } \
                RLH9_ZERO_Y(Q, rl_); Q += rl_; u[Q++] = uint16_t(v_); \
                ST = uint32_t(M.v[ST & 63u][ss_.vh]) | uint32_t(M.r6[ST >> 6][ss_.rc]); \
            } \
        } \
        RLH9_TAIL(Q, QE) } while (0)

#if RLH9_DEC_TBIN_MODE == 2
#define RLH9_TBIN(WIN, SV, USED, POS, Q) \
            const uint64_t tw_ = (WIN) << (USED); \
            const uint32_t re_ = rlut[(((SV) & 15u) << REMB) | uint32_t(tw_ >> (64 - REMB))]; \
            const uint32_t tb_ = tbtbl[(SV) & 2047u]; \
            const uint32_t k_ = tb_ & 15u, t_ = tb_ >> 4; \
            const uint32_t v2_ = uint32_t(tw_ >> (63u - k_)), v1_ = v2_ >> 1, ge_ = uint32_t(v1_ >= t_); \
            const bool is_rem_ = ((SV) <= uint32_t(REMS)); \
            POS += is_rem_ ? (re_ >> 8) : (k_ + ge_); \
            const uint32_t uv_ = is_rem_ ? (re_ & 255u) : (ge_ ? v2_ - t_ : v1_); \
            u[Q] = uint16_t(uv_); u[Q + PS] = uint16_t((SV) - uv_); ++Q;
#elif RLH9_DEC_TBIN_MODE == 1
#define RLH9_TBIN(WIN, SV, USED, POS, Q) \
            uint32_t uv_; \
            if (__builtin_expect((SV) <= uint32_t(REMS), 1)) { \
                const uint32_t re_ = rlut[((SV) << REMB) | uint32_t(((WIN) << (USED)) >> (64 - REMB))]; \
                POS += re_ >> 8; uv_ = re_ & 255u; \
            } else { \
                const uint32_t tb_ = tbtbl[(SV) & 2047u], k_ = tb_ & 15u, t_ = tb_ >> 4; \
                const uint32_t v2_ = uint32_t(((WIN) << (USED)) >> (63u - k_)), v1_ = v2_ >> 1, ge_ = uint32_t(v1_ >= t_); \
                POS += k_ + ge_; uv_ = ge_ ? v2_ - t_ : v1_; \
            } \
            u[Q] = uint16_t(uv_); u[Q + PS] = uint16_t((SV) - uv_); ++Q;
#else
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
#endif

#define RLH9_STEP_S(POS, ST, Q, QE) do { RLH9_HEAD(POS, ST, gs) \
        if (__builtin_expect((e_ & 15u) != 0, 1)) { \
            const uint32_t len_ = e_ & 15u; \
            if (__builtin_expect(int32_t(e_) < 0, 0)) { POS += len_; RLH9_ZERO_S(Q, QE - Q); Q = QE; } \
            else { \
                RLH9_FAST_DECODE(m_, rl_, sv_, ST) \
                const uint32_t used_ = len_ + nm_; POS += used_; \
                RLH9_PUT_S(Q, QE, rl_, w_, sv_, used_, POS) \
                RLH9_UPDATE_ST_FAST(ST) \
            } \
        } else { \
            RLH9_GET_SLOW(ss_) \
            const uint32_t m_ = uint32_t(((w_ << ss_.len) >> 1) >> (63 - ss_.nm)); \
            POS += ss_.len + ss_.nm; \
            if (ss_.eob) { RLH9_ZERO_S(Q, QE - Q); Q = QE; } \
            else { \
                uint32_t rl_ = ss_.rbase + (m_ >> ss_.vm); const uint32_t sv_ = ss_.vbase + (m_ & ss_.vmask); \
                if (rl_ >= QE - Q) { rl_ = QE - Q - 1; bad = true; } \
                RLH9_ZERO_S(Q, rl_); Q += rl_; \
                const uint64_t wr_ = peek(base, POS); \
                RLH9_TBIN(wr_, sv_, 0u, POS, Q) \
                ST = uint32_t(M.v[ST & 63u][ss_.vh]) | uint32_t(M.r6[ST >> 6][ss_.rc]); \
            } \
        } \
        RLH9_TAIL(Q, QE) } while (0)

        while (q0 < e0 && q1 < e1 && q2 < e2 && q3 < e3) { RLH9_STEP_Y(p0, s0, q0, qe0); RLH9_STEP_Y(p1, s1, q1, qe1); RLH9_STEP_S(p2, s2, q2, qe2); RLH9_STEP_S(p3, s3, q3, qe3); }
#if RLH9_DEC_2L_DRAIN == 1
        while (q0 < e0 && q1 < e1) { RLH9_STEP_Y(p0, s0, q0, qe0); RLH9_STEP_Y(p1, s1, q1, qe1); }
        while (q2 < e2 && q3 < e3) { RLH9_STEP_S(p2, s2, q2, qe2); RLH9_STEP_S(p3, s3, q3, qe3); }
#endif
        while (q0 < e0) RLH9_STEP_Y(p0, s0, q0, qe0);
        while (q1 < e1) RLH9_STEP_Y(p1, s1, q1, qe1);
        while (q2 < e2) RLH9_STEP_S(p2, s2, q2, qe2);
        while (q3 < e3) RLH9_STEP_S(p3, s3, q3, qe3);
#undef RLH9_STEP_S
#undef RLH9_TBIN
#undef RLH9_STEP_Y
#undef RLH9_UPDATE_ST_FAST
#undef RLH9_FAST_DECODE
#undef RLH9_TAIL
#undef RLH9_GET_SLOW
#undef RLH9_HEAD
#undef RLH9_ZERO_S
#undef RLH9_ZERO_Y
        return !bad;
    }

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
#if defined(__AVX2__)
    // Transpose two independent 8x8 int16 matrices in-lane inside 8 x __m256i (low128 = matA, high128 = matB)
    inline void tr8x8_avx2(__m256i r[8]) {
        __m256i t0[8], t1[8];
        for (int i = 0; i < 8; i += 2) { t0[i] = _mm256_unpacklo_epi16(r[i], r[i + 1]); t0[i + 1] = _mm256_unpackhi_epi16(r[i], r[i + 1]); }
        for (int i = 0; i < 8; i += 4) {
            t1[i] = _mm256_unpacklo_epi32(t0[i], t0[i + 2]); t1[i + 1] = _mm256_unpackhi_epi32(t0[i], t0[i + 2]);
            t1[i + 2] = _mm256_unpacklo_epi32(t0[i + 1], t0[i + 3]); t1[i + 3] = _mm256_unpackhi_epi32(t0[i + 1], t0[i + 3]);
        }
        for (int i = 0; i < 4; ++i) { r[2 * i] = _mm256_unpacklo_epi64(t1[i], t1[i + 4]); r[2 * i + 1] = _mm256_unpackhi_epi64(t1[i], t1[i + 4]); }
    }
#endif

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

#if defined(__AVX2__)
    // Step 5a: Two independent planes (U in low 128, V in high 128) reconstructed simultaneously in one __m256i wave!
    __attribute__((noinline)) inline void recon_wave8_2p_avx2(const uint16_t* U0, const uint16_t* U1,
                                                              int16_t* P0, int16_t* P1,
                                                              int16_t* top0, int16_t* top1, int W, bool first) {
        if (W < 16) { recon_scalar(U0, P0, top0, W, 8, first); recon_scalar(U1, P1, top1, W, 8, first); return; }
        auto one = [&](const uint16_t* U, int16_t* P, const int16_t* top, int k, int x) {
            const int e = uz(U[size_t(k) * W + x]); int pr;
            if (k == 0 && first) pr = x ? P[x - 1] : 0;
            else { const int b = k ? P[size_t(k - 1) * W + x] : top[x]; pr = x ? med(P[size_t(k) * W + x - 1], b, k ? P[size_t(k - 1) * W + x - 1] : top[x - 1]) : b; }
            P[size_t(k) * W + x] = int16_t(pr + e);
        };
        for (int k = 0; k < 8; ++k) for (int x = 0; x <= 7 - k; ++x) { one(U0, P0, top0, k, x); one(U1, P1, top1, k, x); }
        alignas(32) int16_t a1[16], a2[16];
        for (int k = 0; k < 8; ++k) {
            a1[k] = P0[size_t(k) * W + (7 - k)]; a2[k] = k <= 6 ? P0[size_t(k) * W + (6 - k)] : 0;
            a1[8 + k] = P1[size_t(k) * W + (7 - k)]; a2[8 + k] = k <= 6 ? P1[size_t(k) * W + (6 - k)] : 0;
        }
        __m256i prev1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a1));
        __m256i prev2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a2));
        const __m256i one16 = _mm256_set1_epi16(1), zero = _mm256_setzero_si256();
        int d0 = 8;
        for (; d0 + 8 <= W; d0 += 8) {
            __m256i blk[8];
            for (int k = 0; k < 8; ++k) {
                const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U0 + size_t(k) * W + (d0 - k)));
                const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U1 + size_t(k) * W + (d0 - k)));
                const __m256i v = _mm256_set_m128i(v1, v0);
                blk[k] = _mm256_xor_si256(_mm256_srli_epi16(v, 1), _mm256_sub_epi16(zero, _mm256_and_si256(v, one16)));
            }
            tr8x8_avx2(blk);
            for (int t = 0; t < 8; ++t) {
                const int d = d0 + t; const __m256i a = prev1;
                __m256i b = _mm256_slli_si256(prev1, 2);
                __m256i c = _mm256_slli_si256(prev2, 2);
                if (!first) {
                    const __m256i td = _mm256_set_m128i(_mm_cvtsi32_si128(uint16_t(top1[d])), _mm_cvtsi32_si128(uint16_t(top0[d])));
                    const __m256i tdm = _mm256_set_m128i(_mm_cvtsi32_si128(uint16_t(top1[d - 1])), _mm_cvtsi32_si128(uint16_t(top0[d - 1])));
                    b = _mm256_or_si256(b, td);
                    c = _mm256_or_si256(c, tdm);
                }
                __m256i pred = _mm256_max_epi16(_mm256_min_epi16(a, b), _mm256_min_epi16(_mm256_max_epi16(a, b), _mm256_sub_epi16(_mm256_add_epi16(a, b), c)));
                if (first) pred = _mm256_blend_epi16(pred, a, 0x01);
                const __m256i cur = _mm256_add_epi16(pred, blk[t]); blk[t] = cur; prev2 = prev1; prev1 = cur;
            }
            tr8x8_avx2(blk);
            for (int k = 0; k < 8; ++k) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(P0 + size_t(k) * W + (d0 - k)), _mm256_castsi256_si128(blk[k]));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(P1 + size_t(k) * W + (d0 - k)), _mm256_extracti128_si256(blk[k], 1));
            }
        }
        for (int k = 0; k < 8; ++k) for (int x = d0 - k; x < W; ++x) { one(U0, P0, top0, k, x); one(U1, P1, top1, k, x); }
        std::memcpy(top0, P0 + size_t(7) * W, size_t(W) * sizeof(int16_t));
        std::memcpy(top1, P1 + size_t(7) * W, size_t(W) * sizeof(int16_t));
    }
#endif

#if defined(__SSSE3__)
    // Step 5b + 6b: Fused 3-plane Wave MED + iRCT + RGB24 pack directly to rgb (NO P[3] buffer round-trip!)
    // AVX2 uses 1 x __m256i for U|V + 1 x __m128i for Y; SSE2/SSSE3 uses 3 x __m128i for Y, U, V.
    __attribute__((noinline)) inline void recon_irct_fused8(
        const uint16_t* UY, const uint16_t* UU, const uint16_t* UV,
        int16_t* topY, int16_t* topU, int16_t* topV,
        uint8_t* rgb_sub, int W, bool first) {
        // Fallback for narrow widths
        if (W < 24) {
            int16_t pY[8 * 24], pU[8 * 24], pV[8 * 24];
            recon_scalar(UY, pY, topY, W, 8, first);
            recon_scalar(UU, pU, topU, W, 8, first);
            recon_scalar(UV, pV, topV, W, 8, first);
            for (int r = 0; r < 8; ++r) {
                uint8_t* o = rgb_sub + size_t(r) * W * 3;
                for (int x = 0; x < W; ++x) {
                    const int y = pY[r * W + x], uu = pU[r * W + x];
#if RLH9_RCT == 1
                    const int v = pV[r * W + x] + (uu >> 2), g = y - ((uu + v + 2) >> 2);
#else
                    const int v = pV[r * W + x], g = y - ((uu + v) >> 2);
#endif
                    o[3 * x] = uint8_t(uu + g); o[3 * x + 1] = uint8_t(g); o[3 * x + 2] = uint8_t(v + g);
                }
            }
            return;
        }
        // Small stack buffer for left wedge (cols 0..7) and right wedge (at most 23 cols)
        alignas(16) int16_t wY[8][16], wU[8][16], wV[8][16];
        auto put_px = [&](int k, int x, int y, int uu, int vv) __attribute__((always_inline)) {
            uint8_t* o = rgb_sub + size_t(k) * W * 3 + size_t(x) * 3;
#if RLH9_RCT == 1
            const int v = vv + (uu >> 2), g = y - ((uu + v + 2) >> 2);
#else
            const int v = vv, g = y - ((uu + v) >> 2);
#endif
            o[0] = uint8_t(uu + g); o[1] = uint8_t(g); o[2] = uint8_t(v + g);
        };
        auto one_left = [&](const uint16_t* U, int16_t wbuf[8][16], const int16_t* top, int k, int x) -> int16_t {
            const int e = uz(U[size_t(k) * W + x]); int pr;
            if (k == 0 && first) pr = x ? wbuf[0][x - 1] : 0;
            else { const int b = k ? wbuf[k - 1][x] : top[x]; pr = x ? med(wbuf[k][x - 1], b, k ? wbuf[k - 1][x - 1] : top[x - 1]) : b; }
            const int16_t val = int16_t(pr + e); wbuf[k][x] = val; return val;
        };
        // Left wedge: k = 0..7, x = 0..7-k
        for (int k = 0; k < 8; ++k) {
            for (int x = 0; x <= 7 - k; ++x) {
                const int y = one_left(UY, wY, topY, k, x);
                const int u = one_left(UU, wU, topU, k, x);
                const int v = one_left(UV, wV, topV, k, x);
                put_px(k, x, y, u, v);
            }
        }
        // Save row 7 col 0 for new top (topY/U/V still needed for row 0 at d >= 8, so write to a small staging row or overwrite top[x] once d > x + 1!)
        // Wait: top[d] is read at column d and top[d-1] at column d, so top[d-7] (row 7's column!) is NEVER read again once d >= d0!
        // Because row 7 is at column d0 - 7, and we read top[d0..d0+15] and top[d0-1..d0+14], so top[d0-7..d0+8] overlaps top[d0-1..d0+8]!
        // Therefore keep a tiny 16-element ring delay or separate top_next? Even simpler: store row 7 into top[d0 - 7] ONLY AFTER reading top[d0-1..d0+15] — wait, d0+8 still needs top[d0+7]!
        // So delay writing top by 8 columns (just 1 __m128i register per plane: pending_topY/U/V!!)!
        __m128i pend_topY = _mm_setzero_si128(), pend_topU = _mm_setzero_si128(), pend_topV = _mm_setzero_si128();
        int16_t left_topY = wY[7][0], left_topU = wU[7][0], left_topV = wV[7][0];
        bool has_pend = false;

        alignas(32) int16_t a1Y[8], a2Y[8], a1UV[16], a2UV[16];
        for (int k = 0; k < 8; ++k) {
            a1Y[k] = wY[k][7 - k]; a2Y[k] = k <= 6 ? wY[k][6 - k] : 0;
            a1UV[k] = wU[k][7 - k]; a2UV[k] = k <= 6 ? wU[k][6 - k] : 0;
            a1UV[8 + k] = wV[k][7 - k]; a2UV[8 + k] = k <= 6 ? wV[k][6 - k] : 0;
        }

        const __m128i mR0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5), mG0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1),
                      mB0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1), mR1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1),
                      mG1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10), mB1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1),
                      mR2 = _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1), mG2 = _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1),
                      mB2 = _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15);
        const __m128i m255 = _mm_set1_epi16(255), one16_s = _mm_set1_epi16(1), zero_s = _mm_setzero_si128();
#if RLH9_RCT == 1
        const __m128i k2_s = _mm_set1_epi16(2);
#endif
        auto emit_rgb16 = [&](int k, int col, __m128i y0, __m128i y1, __m128i u0, __m128i u1, __m128i v0, __m128i v1) __attribute__((always_inline)) {
#if RLH9_RCT == 1
            v0 = _mm_add_epi16(v0, _mm_srai_epi16(u0, 2));
            v1 = _mm_add_epi16(v1, _mm_srai_epi16(u1, 2));
            const __m128i g0 = _mm_sub_epi16(y0, _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(u0, v0), k2_s), 2));
            const __m128i g1 = _mm_sub_epi16(y1, _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(u1, v1), k2_s), 2));
#else
            const __m128i g0 = _mm_sub_epi16(y0, _mm_srai_epi16(_mm_add_epi16(u0, v0), 2));
            const __m128i g1 = _mm_sub_epi16(y1, _mm_srai_epi16(_mm_add_epi16(u1, v1), 2));
#endif
            const __m128i r = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(u0, g0), m255), _mm_and_si128(_mm_add_epi16(u1, g1), m255));
            const __m128i g = _mm_packus_epi16(_mm_and_si128(g0, m255), _mm_and_si128(g1, m255));
            const __m128i b = _mm_packus_epi16(_mm_and_si128(_mm_add_epi16(v0, g0), m255), _mm_and_si128(_mm_add_epi16(v1, g1), m255));
            uint8_t* o = rgb_sub + size_t(k) * W * 3 + size_t(col) * 3;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o),      _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR0), _mm_shuffle_epi8(g, mG0)), _mm_shuffle_epi8(b, mB0)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 16), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR1), _mm_shuffle_epi8(g, mG1)), _mm_shuffle_epi8(b, mB1)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 32), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, mR2), _mm_shuffle_epi8(g, mG2)), _mm_shuffle_epi8(b, mB2)));
        };

        __m128i p1Y = _mm_load_si128(reinterpret_cast<const __m128i*>(a1Y));
        __m128i p2Y = _mm_load_si128(reinterpret_cast<const __m128i*>(a2Y));
#if defined(__AVX2__)
        __m256i p1UV = _mm256_load_si256(reinterpret_cast<const __m256i*>(a1UV));
        __m256i p2UV = _mm256_load_si256(reinterpret_cast<const __m256i*>(a2UV));
        const __m256i one16_a = _mm256_set1_epi16(1), zero_a = _mm256_setzero_si256();
#else
        __m128i p1U = _mm_load_si128(reinterpret_cast<const __m128i*>(a1UV));
        __m128i p2U = _mm_load_si128(reinterpret_cast<const __m128i*>(a2UV));
        __m128i p1V = _mm_load_si128(reinterpret_cast<const __m128i*>(a1UV + 8));
        __m128i p2V = _mm_load_si128(reinterpret_cast<const __m128i*>(a2UV + 8));
#endif
        int d0 = 8;
        for (; d0 + 16 <= W; d0 += 16) {
            __m128i bY0[8], bY1[8];
#if defined(__AVX2__)
            __m256i bUV0[8], bUV1[8];
            for (int blk_i = 0; blk_i < 2; ++blk_i) {
                const int db = d0 + blk_i * 8;
                __m128i* bY = blk_i ? bY1 : bY0;
                __m256i* bUV = blk_i ? bUV1 : bUV0;
                for (int k = 0; k < 8; ++k) {
                    const __m128i vy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UY + size_t(k) * W + (db - k)));
                    bY[k] = _mm_xor_si128(_mm_srli_epi16(vy, 1), _mm_sub_epi16(zero_s, _mm_and_si128(vy, one16_s)));
                    const __m128i vu = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UU + size_t(k) * W + (db - k)));
                    const __m128i vv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UV + size_t(k) * W + (db - k)));
                    const __m256i vuv = _mm256_set_m128i(vv, vu);
                    bUV[k] = _mm256_xor_si256(_mm256_srli_epi16(vuv, 1), _mm256_sub_epi16(zero_a, _mm256_and_si256(vuv, one16_a)));
                }
                tr8x8(bY); tr8x8_avx2(bUV);
                for (int t = 0; t < 8; ++t) {
                    const int d = db + t;
                    // Y wave (SSE2)
                    const __m128i aY = p1Y;
                    const __m128i bY_t = _mm_insert_epi16(_mm_slli_si128(p1Y, 2), first ? 0 : topY[d], 0);
                    const __m128i cY_t = _mm_insert_epi16(_mm_slli_si128(p2Y, 2), first ? 0 : topY[d - 1], 0);
                    __m128i prY = _mm_max_epi16(_mm_min_epi16(aY, bY_t), _mm_min_epi16(_mm_max_epi16(aY, bY_t), _mm_sub_epi16(_mm_add_epi16(aY, bY_t), cY_t)));
                    if (first) prY = _mm_insert_epi16(prY, _mm_extract_epi16(aY, 0), 0);
                    const __m128i curY = _mm_add_epi16(prY, bY[t]); bY[t] = curY; p2Y = p1Y; p1Y = curY;
                    // U|V wave (AVX2)
                    const __m256i aUV = p1UV;
                    __m256i bUV_t = _mm256_slli_si256(p1UV, 2);
                    __m256i cUV_t = _mm256_slli_si256(p2UV, 2);
                    if (!first) {
                        bUV_t = _mm256_or_si256(bUV_t, _mm256_set_m128i(_mm_cvtsi32_si128(uint16_t(topV[d])), _mm_cvtsi32_si128(uint16_t(topU[d]))));
                        cUV_t = _mm256_or_si256(cUV_t, _mm256_set_m128i(_mm_cvtsi32_si128(uint16_t(topV[d - 1])), _mm_cvtsi32_si128(uint16_t(topU[d - 1]))));
                    }
                    __m256i prUV = _mm256_max_epi16(_mm256_min_epi16(aUV, bUV_t), _mm256_min_epi16(_mm256_max_epi16(aUV, bUV_t), _mm256_sub_epi16(_mm256_add_epi16(aUV, bUV_t), cUV_t)));
                    if (first) prUV = _mm256_blend_epi16(prUV, aUV, 0x01);
                    const __m256i curUV = _mm256_add_epi16(prUV, bUV[t]); bUV[t] = curUV; p2UV = p1UV; p1UV = curUV;
                }
                tr8x8(bY); tr8x8_avx2(bUV);
            }
            // Write pending top from previous 8-col block (now safe: d0-15..d0-8 will never be read again!)
            if (has_pend) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topY + (d0 - 15)), pend_topY);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topU + (d0 - 15)), pend_topU);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topV + (d0 - 15)), pend_topV);
            } else {
                topY[0] = left_topY; topU[0] = left_topU; topV[0] = left_topV;
                has_pend = true;
            }
            // Row 7 block 0 (cols d0-7..d0) is immediately safe to store into top[d0-7..d0] ONLY AFTER d0+8 finished reading top[d0..d0+15]!
            // Since we just finished BOTH blk 0 and blk 1 (up to d = d0+15), top[d0-7..d0] is safe to store NOW, and blk 1 (cols d0+1..d0+8) is saved as pend_top!
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topY + (d0 - 7)), bY0[7]);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topU + (d0 - 7)), _mm256_castsi256_si128(bUV0[7]));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topV + (d0 - 7)), _mm256_extracti128_si256(bUV0[7], 1));
            pend_topY = bY1[7];
            pend_topU = _mm256_castsi256_si128(bUV1[7]);
            pend_topV = _mm256_extracti128_si256(bUV1[7], 1);
            for (int k = 0; k < 8; ++k) {
                emit_rgb16(k, d0 - k, bY0[k], bY1[k],
                           _mm256_castsi256_si128(bUV0[k]), _mm256_castsi256_si128(bUV1[k]),
                           _mm256_extracti128_si256(bUV0[k], 1), _mm256_extracti128_si256(bUV1[k], 1));
            }
#else
            __m128i bU0[8], bU1[8], bV0[8], bV1[8];
            for (int blk_i = 0; blk_i < 2; ++blk_i) {
                const int db = d0 + blk_i * 8;
                __m128i* bY = blk_i ? bY1 : bY0;
                __m128i* bU = blk_i ? bU1 : bU0;
                __m128i* bV = blk_i ? bV1 : bV0;
                for (int k = 0; k < 8; ++k) {
                    const __m128i vy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UY + size_t(k) * W + (db - k)));
                    const __m128i vu = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UU + size_t(k) * W + (db - k)));
                    const __m128i vv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(UV + size_t(k) * W + (db - k)));
                    bY[k] = _mm_xor_si128(_mm_srli_epi16(vy, 1), _mm_sub_epi16(zero_s, _mm_and_si128(vy, one16_s)));
                    bU[k] = _mm_xor_si128(_mm_srli_epi16(vu, 1), _mm_sub_epi16(zero_s, _mm_and_si128(vu, one16_s)));
                    bV[k] = _mm_xor_si128(_mm_srli_epi16(vv, 1), _mm_sub_epi16(zero_s, _mm_and_si128(vv, one16_s)));
                }
                tr8x8(bY); tr8x8(bU); tr8x8(bV);
                for (int t = 0; t < 8; ++t) {
                    const int d = db + t;
                    auto step_sse = [&](__m128i& p1, __m128i& p2, const int16_t* tp, __m128i& res) __attribute__((always_inline)) {
                        const __m128i a = p1;
                        const __m128i b = _mm_insert_epi16(_mm_slli_si128(p1, 2), first ? 0 : tp[d], 0);
                        const __m128i c = _mm_insert_epi16(_mm_slli_si128(p2, 2), first ? 0 : tp[d - 1], 0);
                        __m128i pr = _mm_max_epi16(_mm_min_epi16(a, b), _mm_min_epi16(_mm_max_epi16(a, b), _mm_sub_epi16(_mm_add_epi16(a, b), c)));
                        if (first) pr = _mm_insert_epi16(pr, _mm_extract_epi16(a, 0), 0);
                        const __m128i cur = _mm_add_epi16(pr, res); res = cur; p2 = p1; p1 = cur;
                    };
                    step_sse(p1Y, p2Y, topY, bY[t]);
                    step_sse(p1U, p2U, topU, bU[t]);
                    step_sse(p1V, p2V, topV, bV[t]);
                }
                tr8x8(bY); tr8x8(bU); tr8x8(bV);
            }
            if (has_pend) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topY + (d0 - 15)), pend_topY);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topU + (d0 - 15)), pend_topU);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(topV + (d0 - 15)), pend_topV);
            } else {
                topY[0] = left_topY; topU[0] = left_topU; topV[0] = left_topV;
                has_pend = true;
            }
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topY + (d0 - 7)), bY0[7]);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topU + (d0 - 7)), bU0[7]);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topV + (d0 - 7)), bV0[7]);
            pend_topY = bY1[7]; pend_topU = bU1[7]; pend_topV = bV1[7];
            for (int k = 0; k < 8; ++k) emit_rgb16(k, d0 - k, bY0[k], bY1[k], bU0[k], bU1[k], bV0[k], bV1[k]);
#endif
        }
        // Right tail: reconstruct columns [d0 - k .. W - 1] for k = 0..7 using prev1/prev2 seed!
        // First extract prev1 (which holds (k, d0 - 1 - k)) into right-tail buffer rY/rU/rV[8][24]
        // Indices in rBuf[k]: col = (d0 - 1 - k) is at offset (7 - k) so all rows align cleanly on col - (d0 - 8)!
        const int base_col = d0 - 8;
        alignas(16) int16_t rY[8][32], rU[8][32], rV[8][32];
        alignas(16) int16_t s1Y[8], s1U[8], s1V[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(s1Y), p1Y);
#if defined(__AVX2__)
        _mm_store_si128(reinterpret_cast<__m128i*>(s1U), _mm256_castsi256_si128(p1UV));
        _mm_store_si128(reinterpret_cast<__m128i*>(s1V), _mm256_extracti128_si256(p1UV, 1));
#else
        _mm_store_si128(reinterpret_cast<__m128i*>(s1U), p1U);
        _mm_store_si128(reinterpret_cast<__m128i*>(s1V), p1V);
#endif
        for (int k = 0; k < 8; ++k) {
            const int x_prev = d0 - 1 - k;
            rY[k][x_prev - base_col] = s1Y[k];
            rU[k][x_prev - base_col] = s1U[k];
            rV[k][x_prev - base_col] = s1V[k];
        }
        // We also need row k-1 at column (d0 - k - 1) for the first tail pixel (k, d0 - k) when k >= 1:
        // Wait: (k-1, d0 - k - 1) is prev2[k-1]! Let's store prev2[0..6] too!
        alignas(16) int16_t s2Y[8], s2U[8], s2V[8];
        _mm_store_si128(reinterpret_cast<__m128i*>(s2Y), p2Y);
#if defined(__AVX2__)
        _mm_store_si128(reinterpret_cast<__m128i*>(s2U), _mm256_castsi256_si128(p2UV));
        _mm_store_si128(reinterpret_cast<__m128i*>(s2V), _mm256_extracti128_si256(p2UV, 1));
#else
        _mm_store_si128(reinterpret_cast<__m128i*>(s2U), p2U);
        _mm_store_si128(reinterpret_cast<__m128i*>(s2V), p2V);
#endif
        for (int k = 0; k < 7; ++k) {
            const int x_prev2 = d0 - 2 - k;
            if (x_prev2 >= base_col) {
                rY[k][x_prev2 - base_col] = s2Y[k];
                rU[k][x_prev2 - base_col] = s2U[k];
                rV[k][x_prev2 - base_col] = s2V[k];
            }
        }
        // Reconstruct tail rows k = 0..7 in order (row 0 reads top[d0-1..W-1], which has NOT been overwritten yet!)
        for (int k = 0; k < 8; ++k) {
            for (int x = d0 - k; x < W; ++x) {
                const int xi = x - base_col;
                auto step_tail = [&](const uint16_t* U, int16_t rbuf[8][32], const int16_t* tp) -> int16_t {
                    const int e = uz(U[size_t(k) * W + x]); int pr;
                    if (k == 0 && first) pr = rbuf[0][xi - 1];
                    else {
                        const int b = k ? rbuf[k - 1][xi] : tp[x];
                        const int c = k ? rbuf[k - 1][xi - 1] : tp[x - 1];
                        pr = med(rbuf[k][xi - 1], b, c);
                    }
                    const int16_t val = int16_t(pr + e); rbuf[k][xi] = val; return val;
                };
                const int y = step_tail(UY, rY, topY);
                const int u = step_tail(UU, rU, topU);
                const int v = step_tail(UV, rV, topV);
                put_px(k, x, y, u, v);
            }
        }
        // Flush pending top[d0-15..d0-8] and right tail of row 7 into top[d0-7..W-1]
        if (has_pend) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topY + (d0 - 15)), pend_topY);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topU + (d0 - 15)), pend_topU);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(topV + (d0 - 15)), pend_topV);
        } else {
            topY[0] = left_topY; topU[0] = left_topU; topV[0] = left_topV;
        }
        for (int x = d0 - 7; x < W; ++x) {
            topY[x] = rY[7][x - base_col];
            topU[x] = rU[7][x - base_col];
            topV[x] = rV[7][x - base_col];
        }
    }
#endif
#endif

    inline void irct_row(const int16_t* Y, const int16_t* U, const int16_t* V, uint8_t* o, int W) {
        int x = 0;
#if defined(RLH9_SSE) && defined(__SSSE3__)
        const __m128i mR0 = _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5), mG0 = _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1),
                      mB0 = _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1), mR1 = _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1),
                      mG1 = _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10), mB1 = _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1),
                      mR2 = _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1), mG2 = _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1),
                      mB2 = _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15);
#if defined(__AVX2__) && RLH9_DEC_IRCT_AVX2 == 1
        const __m256i m255_a = _mm256_set1_epi16(255);
#if RLH9_RCT == 1
        const __m256i k2_a = _mm256_set1_epi16(2);
#endif
        for (; x + 32 <= W; x += 32) {
            __m256i r16[2], g16[2], b16[2];
            for (int h = 0; h < 2; ++h) {
                const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Y + x + 16 * h));
                const __m256i uu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(U + x + 16 * h));
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(V + x + 16 * h));
#if RLH9_RCT == 1
                v = _mm256_add_epi16(v, _mm256_srai_epi16(uu, 2));
                const __m256i g = _mm256_sub_epi16(y, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(uu, v), k2_a), 2));
#else
                const __m256i g = _mm256_sub_epi16(y, _mm256_srai_epi16(_mm256_add_epi16(uu, v), 2));
#endif
                r16[h] = _mm256_and_si256(_mm256_add_epi16(uu, g), m255_a);
                g16[h] = _mm256_and_si256(g, m255_a);
                b16[h] = _mm256_and_si256(_mm256_add_epi16(v, g), m255_a);
            }
            // Pack each 16-pixel half cleanly (note _mm256_packus_epi16 interleaves 128-bit lanes, so use _mm256_permute4x64_epi64(0xD8))
            const __m256i rp = _mm256_permute4x64_epi64(_mm256_packus_epi16(r16[0], r16[1]), 0xD8);
            const __m256i gp = _mm256_permute4x64_epi64(_mm256_packus_epi16(g16[0], g16[1]), 0xD8);
            const __m256i bp = _mm256_permute4x64_epi64(_mm256_packus_epi16(b16[0], b16[1]), 0xD8);
            const __m128i r0 = _mm256_castsi256_si128(rp), g0 = _mm256_castsi256_si128(gp), b0 = _mm256_castsi256_si128(bp);
            const __m128i r1 = _mm256_extracti128_si256(rp, 1), g1 = _mm256_extracti128_si256(gp, 1), b1 = _mm256_extracti128_si256(bp, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x),      _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r0, mR0), _mm_shuffle_epi8(g0, mG0)), _mm_shuffle_epi8(b0, mB0)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 16), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r0, mR1), _mm_shuffle_epi8(g0, mG1)), _mm_shuffle_epi8(b0, mB1)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 32), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r0, mR2), _mm_shuffle_epi8(g0, mG2)), _mm_shuffle_epi8(b0, mB2)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 48), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r1, mR0), _mm_shuffle_epi8(g1, mG0)), _mm_shuffle_epi8(b1, mB0)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 64), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r1, mR1), _mm_shuffle_epi8(g1, mG1)), _mm_shuffle_epi8(b1, mB1)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * x + 80), _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r1, mR2), _mm_shuffle_epi8(g1, mG2)), _mm_shuffle_epi8(b1, mB2)));
        }
#endif
        const __m128i m255 = _mm_set1_epi16(255);
#if RLH9_RCT == 1
        const __m128i k2 = _mm_set1_epi16(2);
#endif
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

    inline size_t decode_pad(int W) { return size_t(W) * HB * 2 * 6 + 64; }
    inline void decode(const uint8_t* base, size_t n, uint8_t* rgb, int W, int H) {
        if (W <= 0 || H <= 0 || uint32_t(W) > MAX_W) throw std::runtime_error("rlh9: bad size");
        const Tables& T = tables();
        std::vector<int16_t> top[3]; for (int p = 0; p < 3; ++p) top[p].assign(size_t(W), 0);
        std::vector<int16_t> P[3];
#if defined(RLH9_SSE) && defined(__AVX2__) && RLH9_DEC_WAVE_MODE == 3
        std::vector<int16_t> P3Y(size_t(8) * W, 0);
#endif
#if !(defined(RLH9_SSE) && defined(__SSSE3__) && RLH9_DEC_WAVE_MODE == 2)
        for (int p = 0; p < 3; ++p) P[p].assign(size_t(8) * W, 0);
#endif
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
#if RLH9_DEC_NO_MEMSET == 0
            std::memset(u.data(), 0, PS * 3 * sizeof(uint16_t));
#endif
            if (!entropy_band(T, base, u.data(), pos, uint32_t(W), uint32_t(nb), sc)) throw std::runtime_error("rlh9: corrupt band");
#if defined(RLH9_SSE) && defined(__AVX2__) && RLH9_DEC_WAVE_MODE == 3
            if (nb == 16 && y0 > 0) {
                
                const uint16_t* UY0 = u.data();
                const uint16_t* UY1 = UY0 + size_t(8) * W;
                const uint16_t* UU0 = UY0 + PS;
                const uint16_t* UU1 = UU0 + size_t(8) * W;
                const uint16_t* UV0 = UU0 + PS;
                const uint16_t* UV1 = UV0 + size_t(8) * W;
                // Pass 1: Y(0..7) + U(0..7)
                recon_wave8_2p_avx2(UY0, UU0, P[0].data(), P[1].data(), top[0].data(), top[1].data(), W, false);
                // Pass 2: V(0..7) + Y(8..15) (independent! Y(8..15) uses top[0] just produced by Pass 1)
                recon_wave8_2p_avx2(UV0, UY1, P[2].data(), P3Y.data(), top[2].data(), top[0].data(), W, false);
                for (int r = 0; r < 8; ++r)
                    irct_row(P[0].data() + size_t(r) * W, P[1].data() + size_t(r) * W, P[2].data() + size_t(r) * W, rgb + size_t(y0 + r) * W * 3, W);
                // Pass 3: U(8..15) + V(8..15)
                recon_wave8_2p_avx2(UU1, UV1, P[1].data(), P[2].data(), top[1].data(), top[2].data(), W, false);
                for (int r = 0; r < 8; ++r)
                    irct_row(P3Y.data() + size_t(r) * W, P[1].data() + size_t(r) * W, P[2].data() + size_t(r) * W, rgb + size_t(y0 + 8 + r) * W * 3, W);
                continue;
            }
#endif
            for (int sub = 0; sub < nb; sub += 8) {
                const int sn = std::min(8, nb - sub); const bool first = y0 + sub == 0;
                const uint16_t* UY = u.data() + size_t(sub) * W;
                const uint16_t* UU = UY + PS;
                (void)UY; (void)UU; const uint16_t* UV = UU + PS; (void)UV;
#if defined(RLH9_SSE) && defined(__SSSE3__) && RLH9_DEC_WAVE_MODE == 2
                if (sn == 8) {
                    recon_irct_fused8(UY, UU, UV, top[0].data(), top[1].data(), top[2].data(), rgb + size_t(y0 + sub) * W * 3, W, first);
                    continue;
                }
                if (P[0].empty()) for (int p = 0; p < 3; ++p) P[p].assign(size_t(8) * W, 0);
#endif
#if defined(RLH9_SSE) && defined(__AVX2__) && RLH9_DEC_WAVE_MODE == 1
                if (sn == 8) {
                    recon_wave8(UY, P[0].data(), top[0].data(), W, first);
                    recon_wave8_2p_avx2(UU, UV, P[1].data(), P[2].data(), top[1].data(), top[2].data(), W, first);
                } else {
                    recon_scalar(UY, P[0].data(), top[0].data(), W, sn, first);
                    recon_scalar(UU, P[1].data(), top[1].data(), W, sn, first);
                    recon_scalar(UV, P[2].data(), top[2].data(), W, sn, first);
                }
#else
                for (int p = 0; p < 3; ++p) {
                    const uint16_t* U = u.data() + size_t(p) * PS + size_t(sub) * W;
#if defined(RLH9_SSE)
                    if (sn == 8) recon_wave8(U, P[p].data(), top[p].data(), W, first); else
#endif
                    recon_scalar(U, P[p].data(), top[p].data(), W, sn, first);
                }
#endif
                for (int r = 0; r < sn; ++r)
                    irct_row(P[0].data() + size_t(r) * W, P[1].data() + size_t(r) * W, P[2].data() + size_t(r) * W, rgb + size_t(y0 + sub + r) * W * 3, W);
            }
        }
    }

    // ---------------- training (tools/rlh9) ----------------
    struct Counts { std::vector<double> c = std::vector<double>(size_t(TOTC) * NSYM, 0.0), rem = std::vector<double>(size_t(NSET) * (REMS + 1) * (REMS + 1), 0.0); };
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
            for (int i = 0; i < n; ++i) q.push({w[i], i});
            int nx = n;
            while (q.size() > 1) { P a = q.top(); q.pop(); P b = q.top(); q.pop(); par[a.second] = nx; par[b.second] = nx; q.push({a.first + b.first, nx++}); }
            int mx = 0; for (int i = 0; i < n; ++i) { int d = 0; for (int j = i; par[j] >= 0; j = par[j]) ++d; L[i] = uint8_t(d); mx = std::max(mx, d); }
            if (mx <= lim) return;
            for (auto& x : w) x = x / 2 + 0.5;
        }
    }
}
