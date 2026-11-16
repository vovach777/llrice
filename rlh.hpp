// rlh.hpp - Run-Length + Quarter-Octave Prefix Huffman (RLH) coder for llrice (LLRICE_ENGINE 6).
// Built on rlgrf::PageBuf with two-slot lazy page reservation (like rlgr_lazy.hpp):
//   - A token is (zero run rl >= 0, non-zero value v > 0), or trailing zeros at flush() with cv = 0.
//   - Reserve(64) is called at the FIRST pixel of a token (the first zero of a run, or v > 0 when rl == 0),
//     and the entire token (<= 64 bits) is written when v becomes known (or at flush()).
//   - Decoder calls reserve(64) at the exact same pixel and reads the entire token at once.
//   - Because cap < 64 implies held <= 1 (since used + cap == 64 * held and 0 <= used <= 63),
//     reserving 64 bits at token start guarantees held <= 2 pages per stream at all times.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "rlgr_fast.hpp"
#include "rlh_tables.h"

namespace rlh {

using rlgrf::PageBuf;

struct Writer {
    PageBuf* pb;
    uint64_t acc = 0;
    int used = 0;
    int64_t cap = 0;
    uint32_t cur = 0, nxt = 0;
    int held = 0;

    explicit Writer(PageBuf* p) : pb(p) {}

    __attribute__((always_inline)) inline void reserve(int n) {
        while (cap < n) {
            const uint32_t p = pb->take();
            assert(held < 2);
            nxt = p;
            cur = held ? cur : p;
            ++held;
            cap += 64;
        }
    }

    __attribute__((always_inline)) inline void store(uint64_t w) {
        pb->w[cur] = w;
        cur = nxt;
        --held;
    }

    // Used by FastRawW (CODER 2 rem_encoder) after reserve(n): 1 <= n <= 32
    __attribute__((always_inline)) inline void put(int n, uint32_t v) {
        assert(cap >= n);
        cap -= n;
        if (__builtin_expect(used + n < 64, 1)) {
            acc = (acc << n) | v;
            used += n;
            return;
        }
        const int rem = used + n - 64;
        store((acc << (n - rem)) | (uint64_t(v) >> rem));
        acc = rem ? (v & ((1u << rem) - 1)) : 0;
        used = rem;
    }

    // Write a whole RLH token (1 <= n <= 64 bits) into space already reserved at token start
    __attribute__((always_inline)) inline void put_tok(int n, uint64_t v) {
        assert(n >= 1 && n <= 64 && cap >= n);
        cap -= n;
        if (__builtin_expect(used + n < 64, 1)) {
            acc = (acc << n) | v;
            used += n;
            return;
        }
        const int rem = used + n - 64;
        store((used ? (acc << (64 - used)) : 0ull) | (v >> rem));
        acc = rem ? (v & ((1ull << rem) - 1)) : 0ull;
        used = rem;
    }

    void flush() {
        if (used) {
            store(acc << (64 - used));
            used = 0;
            acc = 0;
        }
    }
};

struct Reader {
    const uint64_t* w;
    size_t* top;
    size_t n;
    uint64_t c0 = 0, c1 = 0;
    int held = 0, pos = 0;
    int64_t cap = 0;

    Reader(const uint64_t* pages, size_t* t, size_t npages) : w(pages), top(t), n(npages) {}

    __attribute__((always_inline)) inline void take() {
        if (__builtin_expect(*top >= n, 0)) throw std::runtime_error("rlh: out of pages");
        const uint64_t x = w[(*top)++];
        if (held == 0) c0 = x;
        else if (__builtin_expect(held == 1, 1)) c1 = x;
        else throw std::runtime_error("rlh lazy: a third page");
        ++held;
        cap += 64;
    }

    __attribute__((always_inline)) inline void reserve(int k) {
        while (cap < k) take();
    }

    __attribute__((always_inline)) inline uint64_t window() const {
        return pos ? (c0 << pos) | (c1 >> (64 - pos)) : c0;
    }

    __attribute__((always_inline)) inline void skip(int k) {
        pos += k;
        cap -= k;
        if (pos >= 64) {
            pos -= 64;
            c0 = c1;
            c1 = 0;
            --held;
        }
    }
};

// Precomputed value classification for v in 1..2047 (8-bit images after RCT + MED)
struct alignas(8) ValInfo {
    uint16_t vm_val;    // value mantissa bits
    uint8_t cv;         // quarter-octave value class (1, 4..43)
    uint8_t vm_bits;    // number of value mantissa bits (0..8)
    uint8_t ctx_cr0;    // next context when cr == 0 (1..33)
    uint8_t ctx_cr1;    // next context when cr == 1 (1..33)
    uint8_t ctx_cr2;    // next context when cr >= 2 (1..33)
    uint8_t pad;
};

struct CvInfo {
    uint16_t v_base;    // base value for quarter-octave class cv
    uint8_t vm_bits;    // mantissa length for cv (0..8)
    uint8_t pad;
};

struct EncTable {
    // Fast path for rl == 0 (cr == 0, cv in 1..43):
    // c0_code_sh[cv] = uint32_t(code[cv]) << vm_bits[cv]
    // c0_tot_len[cv] = clen[cv] ? (clen[cv] + vm_bits[cv]) : 0
    uint32_t c0_code_sh[NV];
    uint8_t c0_tot_len[NV];
    // General path for cr >= 1 or SYM_ESC (sym = cr * 44 + cv):
    uint16_t code[NS];
    uint8_t len[NS];
};

// Two-level L1/L2-friendly decoder LUT:
//   - Primary 10-bit LUT (1024 entries = 4 KB per context) handles all codewords of length <= 10 (> 98.5% of tokens).
//   - Secondary 2-bit subtable (up to 512 entries = 2 KB per context) handles codewords of length 11..12:
//     when bit 31 == 0 and bit 30 == 1 in lut[top10], bits 0..9 give the base offset in sub[],
//     and sub[base + ((win >> 52) & 3)] contains the exact same fast/slow entry format!
constexpr int LUT1_BITS = 10;
constexpr int LUT2_BITS = MAX_CLEN - LUT1_BITS; // 2

struct DecTable {
    uint32_t lut[1u << LUT1_BITS];
    uint32_t sub[512];
};

static inline uint32_t next_ctx_of(uint32_t cr, uint32_t vk) {
    if (vk == 0) return 0u;
    const uint32_t vk_c = vk < 11u ? vk : 11u;
    const uint32_t cr_c = cr == 0u ? 0u : (cr == 1u ? 1u : 2u);
    return (vk_c - 1u) * 3u + cr_c + 1u;
}

static inline uint32_t select_regime(uint32_t sum_rl, uint32_t hi_vk, uint32_t sum_ctx) {
    if (sum_rl >= 20u) {
        if (hi_vk >= 14u) return 0u;
        if (hi_vk >= 4u || sum_rl >= 96u) return 1u;
        return 2u;
    }
    if (hi_vk >= 16u && sum_rl >= 6u) return 0u;
    if (sum_ctx < 176u) return 3u;
    if (sum_ctx < 266u) return 4u;
    if (sum_ctx < 356u) return 5u;
    if (sum_ctx < 452u) return 6u;
    return 7u;
}

struct GlobalTables {
    ValInfo val_lut[2048];
    CvInfo cv_lut[NV];
    // Channel tables: 0 = luma (g), 1 = chroma (r/b for CODER 3, or rb for CODER 2)
    EncTable enc[2][NREG][NCTX];
    DecTable dec[2][NREG][NCTX];

    static void build_table_pair(const uint8_t packed[PACK_BYTES], const CvInfo cv_lut[NV],
                                 EncTable& et, DecTable& dt) {
        std::memset(&et, 0, sizeof(EncTable));
        std::memset(&dt, 0, sizeof(DecTable));
        uint8_t clen[NS];
        int bl_count[MAX_CLEN + 1] = {};
        for (int s = 0; s < NS; ++s) {
            const uint8_t b = packed[s >> 1];
            const uint8_t l = (s & 1) ? (b >> 4) : (b & 0x0fu);
            clen[s] = l;
            if (l) ++bl_count[l];
        }
        uint32_t next_code[MAX_CLEN + 1] = {};
        uint32_t c = 0;
        for (int bits = 1; bits <= MAX_CLEN; ++bits) {
            c = (c + bl_count[bits - 1]) << 1;
            next_code[bits] = c;
        }
        uint32_t sub_used = 0;
        for (int s = 0; s < NS; ++s) {
            const uint8_t l = clen[s];
            et.len[s] = l;
            if (!l) continue;
            const uint16_t cw = uint16_t(next_code[l]++);
            et.code[s] = cw;

            const uint32_t cr = uint32_t(s / NV);
            const uint32_t cv = uint32_t(s % NV);
            const uint32_t vk = (cv == 0u) ? 0u : (cv == 1u ? 1u : ((cv + 4u) >> 2));
            const uint32_t next_ctx = next_ctx_of(cr, vk);

            uint32_t d_entry = 0;
            if (cr == 0u && cv > 0u) {
                const uint32_t vm_bits = cv_lut[cv].vm_bits;
                const uint32_t v_base = cv_lut[cv].v_base;
                const uint32_t tot_len = uint32_t(l) + vm_bits;
                const uint32_t xor_mask = (uint32_t(cw) << vm_bits) ^ v_base;
                et.c0_code_sh[cv] = uint32_t(cw) << vm_bits;
                et.c0_tot_len[cv] = uint8_t(tot_len);
                d_entry = 0x80000000u | (next_ctx << 25) | (tot_len << 20) | xor_mask;
            } else {
                if (cr == 0u && cv == 0u) {
                    et.c0_code_sh[0] = cw;
                    et.c0_tot_len[0] = l;
                }
                d_entry = (next_ctx << 25) | (cv << 19) | (cr << 14) | (uint32_t(l) << 10) | uint32_t(s);
            }

            if (l <= LUT1_BITS) {
                const uint32_t pad = LUT1_BITS - l;
                const uint32_t base = uint32_t(cw) << pad;
                const uint32_t cnt = 1u << pad;
                for (uint32_t i = 0; i < cnt; ++i) {
                    dt.lut[base + i] = d_entry;
                }
            } else {
                const uint32_t top10 = uint32_t(cw) >> (l - LUT1_BITS);
                uint32_t sub_base = 0;
                if ((dt.lut[top10] & 0x40000000u) == 0u) {
                    sub_base = sub_used;
                    sub_used += (1u << LUT2_BITS);
                    assert(sub_used <= 512u);
                    dt.lut[top10] = 0x40000000u | sub_base;
                } else {
                    sub_base = dt.lut[top10] & 0x3ffu;
                }
                const uint32_t pad = MAX_CLEN - l;
                const uint32_t low_bits = (uint32_t(cw) << pad) & ((1u << LUT2_BITS) - 1u);
                const uint32_t cnt = 1u << pad;
                for (uint32_t i = 0; i < cnt; ++i) {
                    dt.sub[sub_base + low_bits + i] = d_entry;
                }
            }
        }
    }

    GlobalTables() {
        std::memset(val_lut, 0, sizeof(val_lut));
        std::memset(cv_lut, 0, sizeof(cv_lut));
        for (uint32_t v = 1; v < 2048; ++v) {
            const uint32_t b = 32u - __builtin_clz(v);
            uint32_t cv, vm_bits, vm_val, v_base;
            if (b == 1u) {
                cv = 1u;
                vm_bits = 0u;
                vm_val = 0u;
                v_base = 1u;
            } else if (b == 2u) {
                const uint32_t sub = v & 1u;
                cv = 4u + sub;
                vm_bits = 0u;
                vm_val = 0u;
                v_base = 2u + sub;
            } else {
                const uint32_t sub = (v >> (b - 3u)) & 3u;
                cv = 4u * b - 4u + sub;
                vm_bits = b - 3u;
                vm_val = v & ((1u << vm_bits) - 1u);
                v_base = (4u + sub) << vm_bits;
            }
            val_lut[v] = {
                uint16_t(vm_val),
                uint8_t(cv),
                uint8_t(vm_bits),
                uint8_t(next_ctx_of(0u, b)),
                uint8_t(next_ctx_of(1u, b)),
                uint8_t(next_ctx_of(2u, b)),
                0
            };
            cv_lut[cv] = {uint16_t(v_base), uint8_t(vm_bits), 0};
        }

        for (int r = 0; r < NREG; ++r) {
            for (int c = 0; c < NCTX; ++c) {
                build_table_pair(RLH_LEN_G[r][c], cv_lut, enc[0][r][c], dec[0][r][c]);
#if defined(CODER) && CODER == 2
                build_table_pair(RLH_LEN_C2[r][c], cv_lut, enc[1][r][c], dec[1][r][c]);
#else
                build_table_pair(RLH_LEN_C3[r][c], cv_lut, enc[1][r][c], dec[1][r][c]);
#endif
            }
        }
    }
};

inline const GlobalTables g_tables{};

// Track active Encoder / Decoder instances on the current thread so that when constructed
// via standard llrice Encoder(out_pool) / Decoder(pool), the first instance (g_encoder / g_decoder)
// automatically selects channel 0 (luma) and subsequent instances select channel 1 (chroma).
inline int& active_enc_count() {
    thread_local int cnt = 0;
    return cnt;
}
inline int& active_dec_count() {
    thread_local int cnt = 0;
    return cnt;
}

// Variable-length Escape format:
//   SYM_ESC (cr = 0, cv = 0, clen <= 12 bits)
//   + 5-bit cr = bw(rl) (0..26, covering rl up to 67,108,863)
//   + 4-bit vk_hdr: if vk <= 14, vk_hdr = vk (0..14) followed by (vk > 1 ? vk - 1 : 0) mantissa bits;
//                   if vk >= 15 (16-bit images), vk_hdr = 15 followed by 5-bit (vk - 15) + (vk - 1) mantissa bits.
//   + (cr > 1 ? cr - 1 : 0) run mantissa bits + value mantissa bits.
// On 8-bit images, an escape costs only clen + 9 + (cr - 1) + (vk - 1) bits, and always fits in <= 64 bits.
constexpr uint32_t MAX_RUN_ENC = (1u << 26) - 1u;

struct Encoder {
    Writer s;
    uint32_t rl = 0;
    uint32_t ctx = 0;
    uint32_t reg = 1;
    uint32_t sum_rl = 0, hi_vk = 0, sum_ctx = 0, tok_cnt = 0;
    int ch = 0;
    bool in_tok = false;
    bool owns_slot = false;
    const EncTable* reg_base = nullptr;

    explicit Encoder(PageBuf* p, int ch_override = -1) : s(p) {
        if (ch_override >= 0) {
            ch = ch_override > 0 ? 1 : 0;
        } else {
            ch = (active_enc_count()++ > 0) ? 1 : 0;
            owns_slot = true;
        }
        reg_base = g_tables.enc[ch][reg];
    }

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    ~Encoder() {
        if (owns_slot) --active_enc_count();
    }

    __attribute__((noinline)) void update_regime() {
        reg = select_regime(sum_rl, hi_vk, sum_ctx);
        sum_rl = 0;
        hi_vk = 0;
        sum_ctx = 0;
        reg_base = g_tables.enc[ch][reg];
    }

    __attribute__((noinline)) void put_escape(uint32_t esc_rl, uint32_t v, bool is_eof = false) {
        const EncTable& et = reg_base[ctx];
        const uint32_t clen = et.len[0];
        const uint64_t cw = et.code[0];
        const uint32_t cr = esc_rl ? (32u - __builtin_clz(esc_rl)) : 0u;
        const uint32_t vk = v ? (32u - __builtin_clz(v)) : 0u;
        const uint32_t rm_bits = (cr > 1u) ? (cr - 1u) : 0u;
        const uint64_t rm_val = esc_rl & ((1ull << rm_bits) - 1ull);
        const uint32_t vm_bits = (vk > 1u) ? (vk - 1u) : 0u;
        const uint64_t vm_val = v & ((1ull << vm_bits) - 1ull);

        if (__builtin_expect(vk <= 14u, 1)) {
            const uint64_t hdr = (cw << 9) | (uint64_t(cr) << 4) | uint64_t(vk);
            const uint64_t bits = (((hdr << rm_bits) | rm_val) << vm_bits) | vm_val;
            s.put_tok(int(clen + 9u + rm_bits + vm_bits), bits);
        } else {
            const uint64_t hdr = (cw << 14) | (uint64_t(cr) << 9) | (15ull << 5) | uint64_t(vk - 15u);
            const uint64_t bits = (((hdr << rm_bits) | rm_val) << vm_bits) | vm_val;
            s.put_tok(int(clen + 14u + rm_bits + vm_bits), bits);
        }

        if (!is_eof) {
            ctx = next_ctx_of(cr, vk);
            sum_rl += (esc_rl < 256u) ? esc_rl : 256u;
            hi_vk += (ctx >= 16u);
            sum_ctx += ctx;
            if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();
        }
        rl = 0;
        in_tok = false;
    }

    __attribute__((noinline)) void put_with_run(uint32_t v) {
        const uint32_t cur_rl = rl;
        if (__builtin_expect(cur_rl >= (1u << 18) || v >= 2048u, 0)) {
            put_escape(cur_rl, v);
            return;
        }
        const uint32_t cr = 32u - __builtin_clz(cur_rl);
        const ValInfo vi = g_tables.val_lut[v];
        const uint32_t sym = cr * NV + vi.cv;
        const EncTable& et = reg_base[ctx];
        const uint32_t clen = et.len[sym];
        if (__builtin_expect(clen == 0, 0)) {
            put_escape(cur_rl, v);
            return;
        }
        const uint32_t rm_bits = cr - 1u;
        const uint32_t rm_val = cur_rl & ((1u << rm_bits) - 1u);
        const uint64_t cw = et.code[sym];
        const uint64_t bits = (((cw << rm_bits) | rm_val) << vi.vm_bits) | vi.vm_val;
        s.put_tok(int(clen + rm_bits + vi.vm_bits), bits);

        ctx = (cr == 1u) ? vi.ctx_cr1 : vi.ctx_cr2;
        sum_rl += (cur_rl < 256u) ? cur_rl : 256u;
        hi_vk += (ctx >= 16u);
        sum_ctx += ctx;
        if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();
        rl = 0;
        in_tok = false;
    }

    __attribute__((always_inline)) inline void put(uint32_t v) {
        if (v == 0) {
            if (!in_tok) {
                s.reserve(64);
                in_tok = true;
            }
            if (__builtin_expect(++rl == MAX_RUN_ENC, 0)) {
                put_escape(rl, 0);
            }
            return;
        }
        if (__builtin_expect(!in_tok, 1)) {
            s.reserve(64);
            if (__builtin_expect(v < 2048u, 1)) {
                const ValInfo vi = g_tables.val_lut[v];
                const EncTable& et = reg_base[ctx];
                const int tot_len = et.c0_tot_len[vi.cv];
                if (__builtin_expect(tot_len != 0, 1)) {
                    const uint32_t bits = et.c0_code_sh[vi.cv] | vi.vm_val;
                    s.put_tok(tot_len, bits);
                    ctx = vi.ctx_cr0;
                    hi_vk += (ctx >= 16u);
                    sum_ctx += ctx;
                    if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();
                    return;
                }
            }
            put_escape(0, v);
            return;
        }
        put_with_run(v);
    }

    void flush() {
        if (rl > 0) {
            // Space (64 bits) was already reserved when the first zero of this trailing run arrived
            if (rl < (1u << 18)) {
                const uint32_t cr = 32u - __builtin_clz(rl);
                const uint32_t sym = cr * NV + 0u;
                const EncTable& et = reg_base[ctx];
                const uint32_t clen = et.len[sym];
                if (clen != 0) {
                    const uint32_t rm_bits = cr - 1u;
                    const uint32_t rm_val = rl & ((1u << rm_bits) - 1u);
                    const uint64_t cw = et.code[sym];
                    s.put_tok(int(clen + rm_bits), (cw << rm_bits) | rm_val);
                    rl = 0;
                    in_tok = false;
                    s.flush();
                    return;
                }
            }
            put_escape(rl, 0, true);
        }
        s.flush();
    }
};

struct Decoder {
    Reader s;
    uint32_t rl_rem = 0;
    uint32_t pend_v = 0;
    uint32_t ctx = 0;
    uint32_t reg = 1;
    uint32_t sum_rl = 0, hi_vk = 0, sum_ctx = 0, tok_cnt = 0;
    int ch = 0;
    bool in_run = false;
    bool owns_slot = false;
    const DecTable* reg_base = nullptr;

    Decoder(const uint64_t* pages, size_t* top, size_t n, int ch_override = -1) : s(pages, top, n) {
        if (ch_override >= 0) {
            ch = ch_override > 0 ? 1 : 0;
        } else {
            ch = (active_dec_count()++ > 0) ? 1 : 0;
            owns_slot = true;
        }
        reg_base = g_tables.dec[ch][reg];
    }

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    ~Decoder() {
        if (owns_slot) --active_dec_count();
    }

    __attribute__((noinline)) void update_regime() {
        reg = select_regime(sum_rl, hi_vk, sum_ctx);
        sum_rl = 0;
        hi_vk = 0;
        sum_ctx = 0;
        reg_base = g_tables.dec[ch][reg];
    }

    __attribute__((noinline)) uint32_t decode_slow(uint64_t win, uint32_t d) {
        if (d & 0x40000000u) {
            d = reg_base[ctx].sub[(d & 0x3ffu) + uint32_t((win >> (64 - MAX_CLEN)) & ((1u << LUT2_BITS) - 1u))];
            if (int32_t(d) < 0) {
                const int tot_len = (d >> 20) & 31;
                const uint32_t v = uint32_t(win >> (64 - tot_len)) ^ (d & 0xfffffu);
                s.skip(tot_len);
                ctx = (d >> 25) & 63u;
                hi_vk += (ctx >= 16u);
                sum_ctx += ctx;
                if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();
                return v;
            }
        }

        const int clen = (d >> 10) & 15;
        const uint32_t cr = (d >> 14) & 31u;
        const uint32_t cv = (d >> 19) & 63u;
        const uint32_t next_ctx = (d >> 25) & 63u;

        if (__builtin_expect(cr > 0 && cv > 0, 1)) {
            const uint32_t rm_bits = cr - 1u;
            const uint32_t rl = (cr == 1u) ? 1u : ((1u << rm_bits) | uint32_t((win << clen) >> (64 - rm_bits)));
            const CvInfo ci = g_tables.cv_lut[cv];
            const uint32_t vm_bits = ci.vm_bits;
            const uint32_t v = ci.v_base | (vm_bits ? uint32_t((win << (clen + rm_bits)) >> (64 - vm_bits)) : 0u);
            s.skip(int(clen + rm_bits + vm_bits));

            ctx = next_ctx;
            sum_rl += (rl < 256u) ? rl : 256u;
            hi_vk += (ctx >= 16u);
            sum_ctx += ctx;
            if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();

            in_run = true;
            rl_rem = rl;
            pend_v = v;
            return 0;
        }

        if (cv == 0 && cr > 0) {
            // Trailing zero run at EOF
            const uint32_t rm_bits = cr - 1u;
            const uint32_t rl = (cr == 1u) ? 1u : ((1u << rm_bits) | uint32_t((win << clen) >> (64 - rm_bits)));
            s.skip(int(clen + rm_bits));
            if (rl > 1u) {
                in_run = true;
                rl_rem = rl - 1u;
                pend_v = 0;
            }
            return 0;
        }

        // Escape token (cr == 0, cv == 0)
        const uint32_t esc_cr = uint32_t((win << clen) >> (64 - 5));
        uint32_t esc_vk = uint32_t((win << (clen + 5)) >> (64 - 4));
        int hdr_bits = clen + 9;
        if (__builtin_expect(esc_vk == 15u, 0)) {
            esc_vk = 15u + uint32_t((win << (clen + 9)) >> (64 - 5));
            hdr_bits = clen + 14;
        }
        const uint32_t rm_bits = (esc_cr > 1u) ? (esc_cr - 1u) : 0u;
        const uint32_t rl = (esc_cr <= 1u) ? esc_cr : ((1u << rm_bits) | uint32_t((win << hdr_bits) >> (64 - rm_bits)));
        const uint32_t vm_bits = (esc_vk > 1u) ? (esc_vk - 1u) : 0u;
        const uint32_t v = (esc_vk <= 1u) ? esc_vk : ((1u << vm_bits) | uint32_t((win << (hdr_bits + rm_bits)) >> (64 - vm_bits)));
        s.skip(hdr_bits + int(rm_bits + vm_bits));

        ctx = next_ctx_of(esc_cr, esc_vk);
        sum_rl += (rl < 256u) ? rl : 256u;
        hi_vk += (ctx >= 16u);
        sum_ctx += ctx;
        if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();

        if (rl == 0) return v;
        if (v > 0) {
            in_run = true;
            rl_rem = rl;
            pend_v = v;
            return 0;
        }
        if (rl > 1u) {
            in_run = true;
            rl_rem = rl - 1u;
            pend_v = 0;
        }
        return 0;
    }

    __attribute__((always_inline)) inline uint32_t get() {
        if (in_run) {
            if (--rl_rem > 0) return 0;
            in_run = false;
            return pend_v;
        }
        s.reserve(64);
        const uint64_t win = s.window();
        const uint32_t d = reg_base[ctx].lut[win >> (64 - LUT1_BITS)];
        if (__builtin_expect(int32_t(d) < 0, 1)) {
            const int tot_len = (d >> 20) & 31;
            const uint32_t v = uint32_t(win >> (64 - tot_len)) ^ (d & 0xfffffu);
            s.skip(tot_len);
            ctx = (d >> 25) & 63u;
            hi_vk += (ctx >= 16u);
            sum_ctx += ctx;
            if (__builtin_expect((++tok_cnt & 31u) == 0, 0)) update_regime();
            return v;
        }
        return decode_slow(win, d);
    }
};

} // namespace rlh
