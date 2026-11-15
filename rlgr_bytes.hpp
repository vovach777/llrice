// rlgr_bytes.hpp - RLGR streams on byte chunks of variable size cut from one shared byte head (Vladimir's idea).
// No pages, no holes. Every stream (coder or raw bit stream) owns a queue of chunks of the shared buffer; its bits go
// into its own chunks in order, MSB first. The shared head is a plain byte offset: a chunk is cut as
// [head, head + nbytes), head += nbytes. The bit position lives only inside the stream (a 64-bit register).
// Chunks are cut at the llrice reservation points (1 + k before a run header, 64 before every Rice code: the decoder
// knows these sizes), with nbytes = max(ceil(missing bits / 8), G): the bits a stream reserved but did not use stay
// its own and take its next codes, so a full window costs 1 bit, like pages; only the end of each stream (< G bytes
// + 7 bits) is unused. G = RLGR_BYTES_G bytes (8 by default, >= 4). The decoder cuts the same chunks at the same symbols.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "rlgr_fast.hpp"   // RLGR constants and adapt_kr

#ifndef RLGR_BYTES_G
#define RLGR_BYTES_G 8
#endif

namespace rlgrb {
using rlgrf::MAX_K; using rlgrf::UP_K; using rlgrf::DOWN_K; using rlgrf::UP_K_DIRECT_RL; using rlgrf::DOWN_K_DIRECT_RL;
using rlgrf::adapt_kr;
constexpr uint32_t G = RLGR_BYTES_G;
static_assert(G >= 4, "RLGR_BYTES_G < 4: the chunk queue (8) can overflow");

struct Buf {                                        // encoder: the shared byte buffer and its head
    std::vector<uint8_t> b; size_t head = 0;
    inline size_t cut(size_t n) { const size_t s = head; head += n; if (head > b.size()) b.resize(std::max(head, b.size() * 2)); return s; }
};
struct Chunk { size_t p, end; };                    // [p, end) bytes

struct Writer {
    Buf* buf; Chunk q[8]; uint32_t qh = 0, qn = 0;  // chunk queue (front = being written)
    uint64_t acc = 0; int used = 0;                 // bits not yet stored, MSB-aligned to bit `used` from the low end
    int64_t cap = 0;                                // reserved bits not yet put
    explicit Writer(Buf* b) : buf(b) {}
    inline void reserve(int n) {
        if (cap >= n) return;
        const uint64_t bytes = std::max<uint64_t>((uint64_t(n - cap) + 7) >> 3, G);
        if (__builtin_expect(qn == 8, 0)) throw std::runtime_error("rlgr bytes: chunk queue overflow");
        const size_t s = buf->cut(bytes); q[(qh + qn) & 7] = {s, s + bytes}; ++qn; cap += int64_t(bytes) * 8;
    }
    inline void store_byte(uint8_t v) {
        Chunk& c = q[qh & 7]; buf->b[c.p++] = v;
        if (c.p == c.end) { ++qh; --qn; }
    }
    inline void drain() {                           // store whole bytes while 32+ bits wait
        while (used >= 32) {
            Chunk& c = q[qh & 7];
            const uint32_t w = uint32_t(acc >> (used - 32));
            if (c.end - c.p >= 4) { const uint32_t be = __builtin_bswap32(w); std::memcpy(&buf->b[c.p], &be, 4); c.p += 4; if (c.p == c.end) { ++qh; --qn; } }
            else { store_byte(uint8_t(w >> 24)); store_byte(uint8_t(w >> 16)); store_byte(uint8_t(w >> 8)); store_byte(uint8_t(w)); }
            used -= 32;
        }
    }
    inline void put(int n, uint32_t v) {            // 1 <= n <= 32, v < 2^n, reserved
        cap -= n; acc = (acc << n) | v; used += n; if (used >= 32) drain();
    }
    inline void put64(int n, uint64_t v) { if (n > 32) { put(n - 32, uint32_t(v >> 32)); put(32, uint32_t(v)); } else put(n, uint32_t(v)); }
    void flush() {
        drain();
        while (used >= 8) { store_byte(uint8_t(acc >> (used - 8))); used -= 8; }
        if (used) { store_byte(uint8_t(acc << (8 - used))); used = 0; }
        acc = 0;
    }
};

struct In { const uint8_t* b; size_t n; size_t head; };   // decoder: the bytes and the shared head

struct Reader {
    In* in; Chunk q[8]; uint32_t qh = 0, qn = 0; uint64_t win = 0; int valid = 0; int64_t cap = 0;
    explicit Reader(In* i) : in(i) {}
    inline void reserve(int n) {
        if (cap >= n) return;
        const uint64_t bytes = std::max<uint64_t>((uint64_t(n - cap) + 7) >> 3, G);
        if (__builtin_expect(qn == 8, 0)) throw std::runtime_error("rlgr bytes: chunk queue overflow");
        const size_t s = in->head; in->head += bytes;
        if (in->head > in->n) throw std::runtime_error("rlgr bytes: read past the end");
        q[(qh + qn) & 7] = {s, s + bytes}; ++qn; cap += int64_t(bytes) * 8;
    }
    inline void refill() {                          // to 33+ valid bits while the queue has bytes
        while (valid <= 56 && qn) {
            Chunk& c = q[qh & 7];
            if (valid <= 32 && c.end - c.p >= 4) {
                uint32_t be; std::memcpy(&be, in->b + c.p, 4); c.p += 4;
                win |= uint64_t(__builtin_bswap32(be)) << (32 - valid); valid += 32;
            } else { win |= uint64_t(in->b[c.p++]) << (56 - valid); valid += 8; }
            if (c.p == c.end) { ++qh; --qn; }
            if (valid > 32) break;
        }
    }
    inline uint32_t peek32() { if (valid < 32) refill(); return uint32_t(win >> 32); }
    inline void skip(int n) { win <<= n; valid -= n; cap -= n; }
    inline uint32_t get(int n) { if (!n) return 0; if (valid < n) refill(); const uint32_t v = uint32_t(win >> (64 - n)); skip(n); return v; }
};

inline void rice_put(Writer& s, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m;
    if (q >= 32) { s.put(32, 0xffffffffu); s.put(32, x); return; }
    s.put64(int(q + 1 + m), (((2ull << q) - 2) << m) | (x & ((1ull << m) - 1)));
}
inline uint32_t rice_get(Reader& s, uint32_t m) {
    const uint32_t w = s.peek32();
    if (w == 0xffffffffu) { s.skip(32); return s.get(32); }
    const int q = __builtin_clz(~w); s.skip(q + 1);
    return (uint32_t(q) << m) | s.get(int(m));
}

struct Encoder {                                    // llrice RLGR with the llrice reservation points
    Writer s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool inrun = false;
    explicit Encoder(Buf* b) : s(b) {}
    __attribute__((always_inline)) inline void put(uint32_t v) {
        uint32_t k_ = k >> 3; const uint32_t kr_ = kr >> 3;
        if (k_ > 0) {
            if (!inrun) { s.reserve(1 + int(k_)); inrun = true; }
            if (v == 0) {
                if (++rl == (1u << k_)) {
                    s.put(1, 0); rl = 0; k += UP_K; if (k > MAX_K) k = MAX_K;
                    k_ = k >> 3; s.reserve(1 + int(k_));
                }
            } else {
                inrun = false;
                s.put(1, 1); s.put(int(k_), rl);
                s.reserve(64);
                k -= k > DOWN_K ? DOWN_K : k;
                --v; rl = 0; rice_put(s, v, kr_); adapt_kr(kr, v, kr_);
            }
        } else {
            rl = 0; s.reserve(64); rice_put(s, v, kr_); adapt_kr(kr, v, kr_);
            if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        }
    }
    void flush() { if (rl > 0) { s.put(1, 1); s.put(int(k >> 3), rl); rl = 0; } s.flush(); }
};

struct Decoder {
    Reader s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool run_mode = false, fullrl = false;
    explicit Decoder(In* i) : s(i) {}
    inline uint32_t read_rice() { const uint32_t kr_ = kr >> 3; const uint32_t v = rice_get(s, kr_); adapt_kr(kr, v, kr_); return v; }
    inline uint32_t finish_partial_run() { s.reserve(64); k -= k > DOWN_K ? DOWN_K : k; run_mode = false; return read_rice() + 1; }
    inline uint32_t process_run() {
        if (fullrl) {
            if (--rl == 0) { k += UP_K; if (k > MAX_K) k = MAX_K; s.reserve(1 + int(k >> 3)); run_mode = false; }
            return 0;
        }
        if (rl == 0) return finish_partial_run();
        --rl; return 0;
    }
    __attribute__((always_inline)) inline uint32_t get() {
        if (run_mode) return process_run();
        const uint32_t k_ = k >> 3;
        if (k_ > 0) {
            if (!fullrl) s.reserve(1 + int(k_));
            run_mode = true; fullrl = !s.get(1);
            if (fullrl) rl = 1u << k_;
            else { rl = s.get(int(k_)); if (rl == 0) return finish_partial_run(); }
            return process_run();
        }
        s.reserve(64);
        const uint32_t v = read_rice();
        if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        return v;
    }
};
}  // namespace rlgrb
