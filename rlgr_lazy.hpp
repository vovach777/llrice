// rlgr_lazy.hpp - the lazy-reservation RLGR of rlgr_fast.hpp (engine 2) with two page slots instead of a ring.
// With lazy reservation a stream lays ahead at most a run header (1 + k <= 31 bits) or the code it is writing
// (<= 64 bits), and takes a page only when it holds less than that: so it never holds more than two pages, the one
// being written / read and the next one (measured and by this bound). The 4-entry ring with its index masks becomes
// two slots: writer `cur`, `nxt` (page indices), reader `c0`, `c1` (loaded words; c1 = 0 when not loaded, so the
// window needs no count check). Same pages in the same order as rlgr_fast without RLGR_RESERVE64: same files.
#pragma once
#include <cassert>
#include "rlgr_fast.hpp"

namespace rlgrz {
using rlgrf::PageBuf; using rlgrf::MAX_K; using rlgrf::UP_K; using rlgrf::DOWN_K; using rlgrf::UP_K_DIRECT_RL;
using rlgrf::DOWN_K_DIRECT_RL; using rlgrf::adapt_kr;

struct Writer {
    PageBuf* pb; uint64_t acc = 0; int used = 0; int64_t cap = 0; uint32_t cur = 0, nxt = 0; int held = 0;
    explicit Writer(PageBuf* p) : pb(p) {}
    inline void reserve(int n) {
        while (cap < n) {
            const uint32_t p = pb->take();      // the first page held becomes cur, the second nxt (select, no branch)
            assert(held < 2); nxt = p; cur = held ? cur : p; ++held; cap += 64;
        }
    }
    inline void store(uint64_t w) { pb->w[cur] = w; cur = nxt; --held; }   // held <= 2 by the lazy bound (assert in reserve)
    inline void put(int n, uint32_t v) {            // 1 <= n <= 32, v < 2^n, reserved
        cap -= n;
        if (used + n < 64) { acc = (acc << n) | v; used += n; return; }
        const int rem = used + n - 64;              // used >= 32 here
        store((acc << (n - rem)) | (uint64_t(v) >> rem));
        acc = rem ? (v & ((1u << rem) - 1)) : 0; used = rem;
    }
    inline void put64(int n, uint64_t v) {          // 1 <= n <= 64, v < 2^n; takes its page if needed
        reserve(n); cap -= n;
        if (used + n < 64) { acc = (acc << n) | v; used += n; return; }
        const int rem = used + n - 64;
        store((used ? acc << (64 - used) : 0) | (v >> rem));
        acc = rem ? (v & ((1ull << rem) - 1)) : 0; used = rem;
    }
    void flush() { if (used) { store(acc << (64 - used)); used = 0; acc = 0; } }
};

struct Reader {
    const uint64_t* w; size_t* top; size_t n; uint64_t c0 = 0, c1 = 0; int held = 0, pos = 0; int64_t cap = 0;
    Reader(const uint64_t* pages, size_t* t, size_t npages) : w(pages), top(t), n(npages) {}
    inline void take() {
        if (*top >= n) throw std::runtime_error("rlgr: out of pages");
        const uint64_t x = w[(*top)++];
        if (held == 0) c0 = x; else if (__builtin_expect(held == 1, 1)) c1 = x; else throw std::runtime_error("rlgr lazy: a third page");
        ++held; cap += 64;
    }
    inline void reserve(int k) { while (cap < k) take(); }
    inline uint64_t window() const { return pos ? (c0 << pos) | (c1 >> (64 - pos)) : c0; }
    inline void skip(int k) { pos += k; cap -= k; if (pos >= 64) { pos -= 64; c0 = c1; c1 = 0; --held; } }
    inline uint32_t get(int k) { if (!k) return 0; const uint32_t v = uint32_t(window() >> (64 - k)); skip(k); return v; }
};

inline void rice_put(Writer& s, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m;
    if (q >= 32) { s.put64(64, (0xffffffffull << 32) | x); return; }
    s.put64(int(q + 1 + m), (((2ull << q) - 2) << m) | (x & ((1ull << m) - 1)));
}
inline void runend_rice_put(Writer& s, uint32_t k_, uint32_t rl, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m; const uint64_t head = (1ull << k_) | rl;
    if (q < 32 && 1 + k_ + q + 1 + m <= 64)
        s.put64(int(1 + k_ + q + 1 + m), (((head << (q + 1)) | ((2ull << q) - 2)) << m) | (x & ((1ull << m) - 1)));
    else { s.put64(int(1 + k_), head); rice_put(s, x, m); }
}
inline uint32_t rice_get(Reader& s, uint32_t m) {   // the prefix is counted over the bits held; then the code's pages
    uint64_t win; int q;
    for (;;) {
        win = s.window(); q = win == ~0ull ? 64 : __builtin_clzll(~win);
        const int valid = s.cap < 64 ? int(s.cap) : 64;
        if (q < valid || valid >= 32) break;
        s.take();
    }
    if (q >= 32) { s.skip(32); s.reserve(32); return s.get(32); }
    if (s.cap < int64_t(q) + 1 + m) { s.reserve(q + 1 + int(m)); win = s.window(); }
    const uint32_t low = m ? uint32_t((win << (q + 1)) >> (64 - m)) : 0;
    s.skip(q + 1 + int(m)); return (uint32_t(q) << m) | low;
}

struct Encoder {
    Writer s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool inrun = false;
    explicit Encoder(PageBuf* p) : s(p) {}
    __attribute__((always_inline)) inline void put(uint32_t v) {
        uint32_t k_ = k >> 3; const uint32_t kr_ = kr >> 3;
        if (k_ > 0) {
            if (!inrun) { s.reserve(1 + int(k_)); inrun = true; }
            if (v == 0) {
                if (++rl == (1u << k_)) { s.put(1, 0); rl = 0; k += UP_K; if (k > MAX_K) k = MAX_K; k_ = k >> 3; s.reserve(1 + int(k_)); }
            } else {
                inrun = false; k -= k > DOWN_K ? DOWN_K : k;
                --v; runend_rice_put(s, k_, rl, v, kr_); rl = 0; adapt_kr(kr, v, kr_);
            }
        } else {
            rl = 0; rice_put(s, v, kr_); adapt_kr(kr, v, kr_);
            if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        }
    }
    void flush() { if (rl > 0) { s.put(1, 1); s.put(int(k >> 3), rl); rl = 0; } s.flush(); }
};

struct Decoder {
    Reader s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool run_mode = false, fullrl = false;
    Decoder(const uint64_t* pages, size_t* top, size_t n) : s(pages, top, n) {}
    inline uint32_t read_rice() { const uint32_t kr_ = kr >> 3; const uint32_t v = rice_get(s, kr_); adapt_kr(kr, v, kr_); return v; }
    inline uint32_t finish_partial_run() { k -= k > DOWN_K ? DOWN_K : k; run_mode = false; return read_rice() + 1; }
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
            const uint64_t win = s.window(); run_mode = true; fullrl = !(win >> 63);
            if (fullrl) { rl = 1u << k_; s.skip(1); }
            else { rl = uint32_t((win << 1) >> (64 - k_)); s.skip(1 + int(k_)); if (rl == 0) return finish_partial_run(); }
            return process_run();
        }
        const uint32_t v = read_rice();
        if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        return v;
    }
};
}  // namespace rlgrz
