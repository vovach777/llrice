// rlgr_fast.hpp - interleaved RLGR for palic2r: the llrice RLGR coder (same adaptation, same reservation points)
// on a lean shared page buffer. Many bit streams share one buffer of 64-bit pages; a stream takes the next free
// page only at its reservation points (reserve(n): make sure n more bits fit), never while writing, so the
// decoder, which reserves at the same points of the same symbol sequence, takes the same pages in the same order.
// Differences from llrice: the page buffer is one preallocated array (no vector growth, no PagePool), the
// reserved pages of a stream sit in a 4-entry ring of indices (writer) or of loaded words (reader), the reader
// peeks from a two-word window, the Rice prefix is decoded with one count-leading-ones.
// Reservation (default, "lazy"): only the run header is laid in advance (1 + k bits when a run starts and after a
// full window: the encoder writes it at the end of the run, the decoder reads it at the start). Every other code
// takes a page when its first bit is written / read, at the same symbol on both sides. -DRLGR_RESERVE64: the llrice
// rule (also 64 bits before every Rice code), byte-identical with the llrice classes.
// Bits are MSB first inside a page. Pages are written as uint64 in host order.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace rlgrf {
#if defined(RLGR_STATS)                             // research counters (../hlc/research/plr_onestream_estimate.py)
inline uint64_t g_payload = 0, g_waste_full = 0, g_nfull = 0, g_waste_flush = 0, g_nruns = 0;
// simulation of variant (b): one bit head for all codes, run headers in holes of 1 + k; a full window whose hole
// still ends at the head gives its k unused bits back ("shrink"), otherwise they are lost
inline uint64_t g_head = 0, g_nshrink = 0, g_lost_b = 0;
#define RLGR_ST(x) x
#else
#define RLGR_ST(x)
#endif
constexpr uint32_t MAX_K = 240, UP_K = 5, DOWN_K = 9, UP_K_DIRECT_RL = 2, DOWN_K_DIRECT_RL = 3, DOWN_K_RICE = 2;   // llrice

struct PageBuf {                                    // encoder: grows on demand; decoder: the pages of the file
    std::vector<uint64_t> w; size_t top = 0;
    inline uint32_t take() { if (top == w.size()) w.resize(w.size() * 2 + 1024); w[top] = 0; return uint32_t(top++); }
};

struct Writer {
    PageBuf* pb; uint64_t acc = 0; int used = 0; int64_t cap = 0; uint32_t q[4]; uint32_t head = 0, cnt = 0;
    explicit Writer(PageBuf* p) : pb(p) {}
    inline void reserve(int n) { while (cap < n) { q[(head + cnt) & 3] = pb->take(); ++cnt; cap += 64; } }
    inline void put(int n, uint32_t v) {            // 1 <= n <= 32, v < 2^n; the bits must be reserved
        RLGR_ST(g_payload += uint64_t(n); g_head += uint64_t(n);) cap -= n;
        if (used + n < 64) { acc = (acc << n) | v; used += n; return; }
        const int rem = used + n - 64;              // used >= 32 here, so 64 - used <= 32
        const uint64_t w = (acc << (n - rem)) | (uint64_t(v) >> rem);
        pb->w[q[head & 3]] = w; ++head; --cnt;
        acc = rem ? (v & ((1u << rem) - 1)) : 0; used = rem;
    }
    inline void put64(int n, uint64_t v) {          // 1 <= n <= 64, v < 2^n
#if !defined(RLGR_RESERVE64)
        reserve(n);                                 // lazy: a page is taken when its first bit is written
#endif
        RLGR_ST(g_payload += uint64_t(n); g_head += uint64_t(n);) cap -= n;
        if (used + n < 64) { acc = (acc << n) | v; used += n; return; }
        const int rem = used + n - 64;
        const uint64_t hi = used ? acc << (64 - used) : 0;
        pb->w[q[head & 3]] = hi | (v >> rem); ++head; --cnt;
        acc = rem ? (v & ((1ull << rem) - 1)) : 0; used = rem;
    }
    void flush() { if (used) { pb->w[q[head & 3]] = acc << (64 - used); ++head; --cnt; used = 0; acc = 0; } }
};

struct Reader {
    const uint64_t* w; size_t* top; size_t n; uint64_t r[4] = {0, 0, 0, 0}; uint32_t head = 0, cnt = 0; int pos = 0; int64_t cap = 0;
    Reader(const uint64_t* pages, size_t* t, size_t npages) : w(pages), top(t), n(npages) {}
    inline void reserve(int k) {
        while (cap < k) { if (*top >= n) throw std::runtime_error("rlgr: out of pages"); r[(head + cnt) & 3] = w[(*top)++]; ++cnt; cap += 64; }
    }
    inline uint64_t window() const {                // the next 64 bits (needs cap >= what is read)
        const uint64_t cur = r[head & 3];
        if (!pos) return cur;
        const uint64_t nxt = cnt > 1 ? r[(head + 1) & 3] : 0;
        return (cur << pos) | (nxt >> (64 - pos));
    }
    inline void take() { if (*top >= n) throw std::runtime_error("rlgr: out of pages"); r[(head + cnt) & 3] = w[(*top)++]; ++cnt; cap += 64; }
    inline void skip(int k) { pos += k; cap -= k; if (pos >= 64) { pos -= 64; ++head; --cnt; } }
    inline uint32_t get(int k) { if (!k) return 0; const uint32_t v = uint32_t(window() >> (64 - k)); skip(k); return v; }
};

template <class W> inline void rice_put(W& s, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m;
    if (q >= 32) { s.put64(64, (0xffffffffull << 32) | x); return; }
    s.put64(int(q + 1 + m), (((2ull << q) - 2) << m) | (x & ((1ull << m) - 1)));   // q + 1 + m <= 63
}
// run end "1, rl (k_ bits)" followed by the Rice code of v, one write when it fits in 64 bits
inline void runend_rice_put(Writer& s, uint32_t k_, uint32_t rl, uint32_t x, uint32_t m) {
    const uint32_t q = x >> m;
    const uint64_t head = (1ull << k_) | rl;
    if (q < 32 && 1 + k_ + q + 1 + m <= 64)
        s.put64(int(1 + k_ + q + 1 + m), (((head << (q + 1)) | ((2ull << q) - 2)) << m) | (x & ((1ull << m) - 1)));
    else { s.put64(int(1 + k_), head); rice_put(s, x, m); }
}
#if defined(RLGR_RESERVE64)
inline uint32_t rice_get(Reader& s, uint32_t m) {
    const uint64_t win = s.window(); const int q = win == ~0ull ? 64 : __builtin_clzll(~win);   // leading ones
    if (q >= 32) { s.skip(32); return s.get(32); }
    const uint32_t low = m ? uint32_t((win << (q + 1)) >> (64 - m)) : 0;   // q + 1 + m <= 63: all in the window
    s.skip(q + 1 + int(m)); return (uint32_t(q) << m) | low;
}
#else
// Lazy: no worst case laid in advance. The prefix is counted only over the bits the stream already holds (cap); when
// they are all ones the code goes on into the next page, which the encoder took while writing this same code, so
// the decoder takes it now. Then exactly q + 1 + m bits are made available: the same pages as the encoder took.
inline uint32_t rice_get(Reader& s, uint32_t m) {
    uint64_t win; int q;
    for (;;) {
        win = s.window(); q = win == ~0ull ? 64 : __builtin_clzll(~win);
        const int valid = s.cap < 64 ? int(s.cap) : 64;
        if (q < valid || valid >= 32) break;        // zero seen, or 32 ones seen (escape)
        s.take();
    }
    if (q >= 32) { s.skip(32); s.reserve(32); return s.get(32); }
    if (s.cap < int64_t(q) + 1 + m) { s.reserve(q + 1 + int(m)); win = s.window(); }
    const uint32_t low = m ? uint32_t((win << (q + 1)) >> (64 - m)) : 0;
    s.skip(q + 1 + int(m)); return (uint32_t(q) << m) | low;
}
#endif
inline void adapt_kr(uint32_t& kr, uint32_t v, uint32_t kr_) {
    const uint32_t vk = v >> kr_;
    if (vk == 0) kr -= kr > DOWN_K_RICE ? DOWN_K_RICE : kr;
    else if (vk != 1) { kr += vk; if (kr > MAX_K) kr = MAX_K; }
}

struct Encoder {                                    // RLGR::Encoder of llrice, USE_MODEL_K off
    Writer s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool inrun = false; RLGR_ST(uint64_t hole_end = 0;)
    explicit Encoder(PageBuf* p) : s(p) {}
    __attribute__((always_inline)) inline void put(uint32_t v) {
        uint32_t k_ = k >> 3; const uint32_t kr_ = kr >> 3;
        if (k_ > 0) {
            if (!inrun) { s.reserve(1 + int(k_)); inrun = true; RLGR_ST(++g_nruns; g_head += 1 + k_; hole_end = g_head;) }
            if (v == 0) {
                if (++rl == (1u << k_)) {
                    RLGR_ST(g_waste_full += k_; ++g_nfull;)
                    s.put(1, 0); rl = 0; k += UP_K; if (k > MAX_K) k = MAX_K;
                    RLGR_ST(g_head -= 1; if (g_head == hole_end) { g_head -= k_; ++g_nshrink; } else g_lost_b += k_;)
                    k_ = k >> 3; s.reserve(1 + int(k_));
                    RLGR_ST(g_head += 1 + k_; hole_end = g_head;)
                }
            } else {
                inrun = false;
#if defined(RLGR_RESERVE64)
                s.reserve(65 + int(k_));                // = put the run end, then reserve(64): same pages
#endif
                k -= k > DOWN_K ? DOWN_K : k;
                --v; runend_rice_put(s, k_, rl, v, kr_); rl = 0; adapt_kr(kr, v, kr_); RLGR_ST(g_head -= 1 + k_;)
            }
        } else {
            rl = 0;
#if defined(RLGR_RESERVE64)
            s.reserve(64);
#endif
            rice_put(s, v, kr_); adapt_kr(kr, v, kr_);
            if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        }
    }
    void flush() {
        if (rl > 0) { s.put(1, 1); s.put(int(k >> 3), rl); rl = 0; RLGR_ST(g_head -= 1 + (k >> 3);) }
        RLGR_ST(else if (inrun) { g_waste_flush += 1 + (k >> 3); if (g_head == hole_end) g_head -= 1 + (k >> 3); else g_lost_b += 1 + (k >> 3); })
        s.flush(); }
};

struct Decoder {                                    // RLGR::Decoder of llrice
    Reader s; uint32_t rl = 0, k = 1 << 3, kr = 1 << 3; bool run_mode = false, fullrl = false;
    Decoder(const uint64_t* pages, size_t* top, size_t n) : s(pages, top, n) {}
    inline uint32_t read_rice() { const uint32_t kr_ = kr >> 3; const uint32_t v = rice_get(s, kr_); adapt_kr(kr, v, kr_); return v; }
    inline uint32_t finish_partial_run() {
#if defined(RLGR_RESERVE64)
        s.reserve(64);
#endif
        k -= k > DOWN_K ? DOWN_K : k; run_mode = false; return read_rice() + 1; }
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
#if defined(RLGR_RESERVE64)
        s.reserve(64);
#endif
        const uint32_t v = read_rice();
        if (v == 0) { k += UP_K_DIRECT_RL; if (k > MAX_K) k = MAX_K; } else k -= k > DOWN_K_DIRECT_RL ? DOWN_K_DIRECT_RL : k;
        return v;
    }
};
}  // namespace rlgrf
