// rlgr_lane2.hpp - rlgr_lane with holes of several pieces: the unused bits of a full window are reused by the next
// hole of the same coder even when other coders wrote after it.
// In rlgr_lane a full window gives its k unused bits back only if its hole still ends at the head; with several coders
// on one lane (g, r, b) somebody has almost always written after it, and the k bits are lost (+9..14% on screens).
// Here a hole is up to LANE2_PIECES pieces (8 by default), piece 0 first. After a full window the flag (the first bit
// of piece 0) stays 0 and the other bits of the hole start the next hole, which needs 1 + k' bits (k' = k or k + 1):
//  - the last piece ends at the head: it grows at the head (the old give-back, now for any piece);
//  - otherwise the missing bits become a new piece at the head;
//  - all pieces in use: the last piece moves to the head (its old bits are lost; with 8 pieces this practically never
//    happens: the size equals the paged engines).
// The value of a hole (flag, then rl in k bits) is split over the pieces: high bits in piece 0, low bits in the last.
// The decoder moves its head the same way and takes the same decisions at the same symbol. Lane / LaneR / Rice codes
// come from rlgr_lane.hpp.
#pragma once
#include "rlgr_lane.hpp"

namespace rlgrl2 {
using namespace rlgrl;

#ifndef LANE2_PIECES
#define LANE2_PIECES 8
#endif
struct Hole {                                       // up to LANE2_PIECES pieces, piece 0 first in the value
    static constexpr int P = LANE2_PIECES;
    uint64_t pos[P]; uint32_t len[P]; int n = 0; uint32_t tot = 0;   // tot = the sum of len
    inline void fresh(uint64_t& head, uint32_t sz) { pos[0] = head; len[0] = sz; n = 1; tot = sz; head += sz; }
    // full window: the flag bit (first bit of piece 0) is consumed; the rest + bits at the head become `need` bits
    inline void next(uint64_t& head, uint32_t need) {
        ++pos[0]; --tot;
        if (__builtin_expect(--len[0] == 0, 0)) { for (int i = 1; i < n; ++i) { pos[i - 1] = pos[i]; len[i - 1] = len[i]; } --n; }
        const uint32_t ext = need - tot; tot = need;
        if (pos[n - 1] + len[n - 1] == head) { len[n - 1] += ext; head += ext; return; }   // the last piece grows
        if (__builtin_expect(n == P, 0)) { --n; pos[n] = head; len[n] += ext; head += len[n]; ++n; return; }   // no room: the last piece moves to the head (its old bits lost)
        pos[n] = head; len[n] = ext; head += ext; ++n;
    }
};

struct Encoder {
    Lane* s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool inrun = false; Hole h;
    explicit Encoder(Lane* l) : s(l) {}
    inline void fill(uint32_t k_, uint32_t v) {     // v = 1 + k_ bits: flag, rl; the last piece takes the low bits
        (void)k_;
        if (h.n == 1) { s->at(h.pos[0], int(h.len[0]), v); return; }
        for (int i = h.n - 1; i >= 0; --i) { s->at(h.pos[i], int(h.len[i]), v & ((1u << h.len[i]) - 1)); v >>= h.len[i]; }
    }
    __attribute__((always_inline)) inline void put(uint32_t v) {
        uint32_t k_ = k >> 3; const uint32_t kr_ = kr >> 3;
        if (k_ > 0) {
            if (!inrun) { s->grow(); h.fresh(s->head, 1 + k_); inrun = true; }
            if (v == 0) {
                if (++rl == (1u << k_)) {           // full window: flag 0 = the hole's first bit stays 0
                    rl = 0; k += UP_K; if (k > MAX_K) k = MAX_K; k_ = k >> 3;
                    s->grow(); h.next(s->head, 1 + k_);
                }
            } else {
                inrun = false; fill(k_, (1u << k_) | rl);
                k -= k > DOWN_K ? DOWN_K : k;
                --v; rl = 0; rice_put(*s, v, kr_); adapt_kr(kr, v, kr_);
            }
        } else {
            rl = 0; rice_put(*s, v, kr_); adapt_kr(kr, v, kr_);
            if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        }
    }
    void flush() { if (inrun && rl > 0) { fill(k >> 3, (1u << (k >> 3)) | rl); rl = 0; } }
};

struct Decoder {
    LaneR* s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool run_mode = false, fullrl = false; Hole h;
    explicit Decoder(LaneR* l) : s(l) {}
    inline uint32_t read_hole() const {
        uint32_t v = uint32_t(s->peek_at(h.pos[0]) >> (64 - h.len[0]));
        for (int i = 1; i < h.n; ++i) v = (v << h.len[i]) | uint32_t(s->peek_at(h.pos[i]) >> (64 - h.len[i]));
        return v;
    }
    inline uint32_t read_rice() { const uint32_t kr_ = kr >> 3; const uint32_t v = rice_get(*s, kr_); adapt_kr(kr, v, kr_); return v; }
    inline uint32_t finish_partial_run() { k -= k > DOWN_K ? DOWN_K : k; run_mode = false; return read_rice() + 1; }
    inline uint32_t process_run() {
        if (fullrl) {
            if (--rl == 0) {                        // end of a full window: the same hole rule as the encoder
                k += UP_K; if (k > MAX_K) k = MAX_K; h.next(s->head, 1 + (k >> 3)); run_mode = false;
            }
            return 0;
        }
        if (rl == 0) return finish_partial_run();
        --rl; return 0;
    }
    __attribute__((always_inline)) inline uint32_t get() {
        if (run_mode) return process_run();
        const uint32_t k_ = k >> 3;
        if (k_ > 0) {
            if (!fullrl) h.fresh(s->head, 1 + k_);   // a run starts (after a full window the hole is already there)
            const uint32_t v = read_hole(); run_mode = true; fullrl = !(v >> k_);
            if (fullrl) rl = 1u << k_;
            else { rl = v & ((1u << k_) - 1); if (rl == 0) return finish_partial_run(); }
            return process_run();
        }
        const uint32_t v = read_rice();
        if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        return v;
    }
};
}  // namespace rlgrl2
