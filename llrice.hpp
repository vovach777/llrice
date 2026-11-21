#pragma once
#include <functional>
#include <cassert>
#include <algorithm>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <type_traits>
#include <limits>
#include <type_traits>
#include <tuple>
#include "pool.hpp"
#include "bitstream.hpp"
#include "rlgr.hpp"
#include "image.hpp"
// LLRICE_ENGINE - the bit stream engine of the RLGR coders (compile time, like CODER):
//   0 - the original RLGR / BitStream / PagePool classes;
//   1 - rlgr_fast.hpp: lean page buffer, 4-entry page rings, Rice prefix by count-leading-ones, with the
//       llrice reservation rule (RLGR_RESERVE64): the same files as 0 (signature rev '1');
//   2 - lazy reservation ("holes"): only a run header is laid ahead (1 + k bits), every other code takes a page when its
//       first bit is written, so a stream holds at most two pages: rlgr_lazy.hpp, two page slots instead of a ring;
//       a new format (rev '2');
//   3 - rlgr_lane.hpp: one bit lane for all the coders of the image; every code is written at the head the moment it is
//       known, a run header gets a hole of 1 + k bits that is filled at the end of the run, a full window gives its
//       k unused bits back when nothing was written after its hole; no pages; a new format (rev '3');
//   4 - rlgr_lane2.hpp: the lane of 3 with holes of up to 8 pieces: the unused bits of a full window start the next
//       hole of the same coder even if other coders wrote after it; the size of 1/2, no pages (rev '4');
//   5 - rlgr_bytes.hpp: every stream owns byte chunks of variable size cut from one shared byte head at the llrice
//       reservation points; no pages, no holes, the bit position only inside the stream (rev '5').
//   9 - rlh9.hpp: RLH8 superscalar photo codec R8F1 (8-row bands, 3 plane lanes decoded token-interleaved, implicit
//       RLGR-state context, static Huffman); 8-bit RGB only; its own body after the llrice header (rev '9').
// LLRICE_FAST=0 is the old spelling of LLRICE_ENGINE=0. USE_SIMPLE_RLGR, USE_MODEL_K and USE_NEW_LINE need 0.
#ifndef LLRICE_ENGINE
#if (defined(LLRICE_FAST) && LLRICE_FAST == 0) || defined(USE_SIMPLE_RLGR) || defined(USE_MODEL_K) || defined(USE_NEW_LINE)
#define LLRICE_ENGINE 0
#else
#define LLRICE_ENGINE 7   // RLH7 fast: better ratio than RLGR on screens and photos, and faster
#endif
#endif
#if LLRICE_ENGINE == 1 && !defined(RLGR_RESERVE64)
#define RLGR_RESERVE64   // the llrice reservation points: same pages, same file
#endif
#if LLRICE_ENGINE == 2 && defined(RLGR_RESERVE64)
#error "LLRICE_ENGINE 2 is the lazy reservation: do not define RLGR_RESERVE64"
#endif
#if LLRICE_ENGINE == 1
#include "rlgr_fast.hpp"
#elif LLRICE_ENGINE == 2
#include "rlgr_lazy.hpp"
#elif LLRICE_ENGINE == 3
#include "rlgr_lane.hpp"
#elif LLRICE_ENGINE == 4
#include "rlgr_lane2.hpp"
#elif LLRICE_ENGINE == 5
#include "rlgr_bytes.hpp"
#endif

//#define USE_YCoCg
//#define USE_SIMPLE_RLGR
#define CODER_GRB    0
#define CODER_G_RB   1
#define CODER_G_RB_SUM 2
#define CODER_G_R_B 3
#ifndef CODER
#define CODER CODER_G_R_B
#endif
#if LLRICE_ENGINE == 6
#include "rlh.hpp"
#endif
#if LLRICE_ENGINE == 7
#define RLH7_MODE 0
#include "rlh7.hpp"
#endif
#if LLRICE_ENGINE == 8
#define RLH7_MODE 1
#include "rlh7.hpp"
#endif
#if LLRICE_ENGINE == 9
#include "rlgr_fast.hpp"   // only so that the RLGR paths keep compiling; ENGINE 9 does not use them
#include "rlh9.hpp"
#endif
// CODER_G_RB_SUM: r in 0..rb (rb + 1 values). TRUNC_SPLIT 1: truncated binary, k = floor(log2(rb + 1)) bits if
// r < u = 2^(k+1) - (rb + 1), else k + 1 bits of r + u (no code is wasted). 0: bit_width(rb) bits (old format).
#ifndef TRUNC_SPLIT
#define TRUNC_SPLIT 1
#endif
//#define USE_NEW_LINE
// LLRICE_RCT_FNBLI 1: the colour transform of fNBLI / PALIC: Y = G + ((U + V + 2) >> 2), then V -= U >> 2 (V stays -255..255).
#ifndef LLRICE_RCT_FNBLI
#define LLRICE_RCT_FNBLI 0
#endif
#define LLRICE_RCT_RND (LLRICE_RCT_FNBLI ? 2 : 0)
namespace llrice
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = LLRICE_ENGINE >= 2 ? uint8_t('0' + LLRICE_ENGINE) : uint8_t('1');   // new formats: other signatures
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

    // The first line of a .llr file: the program name and the build parameters that change the format, e.g.
    // "llrice ENGINE=1 CODER=3 RCT=fnbli PRESET=screens\n". A decoder only accepts a file of its own build: the format is
    // bound to the build while it is not stable.
    inline std::string build_tag() {
        std::string t = "llrice ENGINE=" + std::to_string(LLRICE_ENGINE) + " CODER=" + std::to_string(CODER);
#if LLRICE_ENGINE == 9
        t += RLH9_RCT ? " RCT=fnbli" : " RCT=rct";
#else
        t += LLRICE_RCT_FNBLI ? " RCT=fnbli" : " RCT=rct";
#endif
#if LLRICE_ENGINE <= 5
        t += LLRICE_RLGR_PRESET == 1 ? " PRESET=screens" : " PRESET=clic";
#endif
#if defined(RLGR_POOL3)
        t += " POOL3";
#endif
#if TRUNC_SPLIT != 1
        t += " TRUNC_SPLIT=" + std::to_string(TRUNC_SPLIT);
#endif
        return t;
    }

    // stream classes per page store: PagePool -> original llrice classes, rlgrf::PageBuf / FastIn -> rlgr_fast
    template <typename Pool> struct Coders;
    template <> struct Coders<PagePool> {
#ifdef USE_SIMPLE_RLGR
        using Enc = RLGR::Simple::Encoder; using Dec = RLGR::Simple::Decoder;
#else
        using Enc = RLGR::Encoder; using Dec = RLGR::Decoder;
#endif
        using RawW = BitStream::Writer; using RawR = BitStream::Reader;
    };
#if LLRICE_ENGINE == 1 || LLRICE_ENGINE == 2 || LLRICE_ENGINE == 6 || LLRICE_ENGINE == 7 || LLRICE_ENGINE == 8 || LLRICE_ENGINE == 9
#if LLRICE_ENGINE == 1 || LLRICE_ENGINE == 9   // 9: the RLGR paths only have to compile
    namespace fast_ns = rlgrf;
#elif LLRICE_ENGINE == 2
    namespace fast_ns = rlgrz;
#elif LLRICE_ENGINE == 6
    namespace fast_ns = rlh;
#else
    namespace fast_ns = rlh7;
#endif
    struct FastIn { const uint64_t* p; size_t top; size_t n; };   // decoder side: the pages and the next free one
    struct FastEnc : fast_ns::Encoder { explicit FastEnc(rlgrf::PageBuf& pb) : fast_ns::Encoder(&pb) {} };
    struct FastDec : fast_ns::Decoder { explicit FastDec(FastIn& in) : fast_ns::Decoder(in.p, &in.top, in.n) {} };
    struct FastRawW {
        fast_ns::Writer w; explicit FastRawW(rlgrf::PageBuf& pb) : w(&pb) {}
        inline void reserve(int n) { w.reserve(n); }
        inline void put_bits(uint32_t n, uint32_t v) { if (n) w.put(int(n), v); }
        inline void flush() { w.flush(); }
    };
    struct FastRawR {
        fast_ns::Reader r; explicit FastRawR(FastIn& in) : r(in.p, &in.top, in.n) {}
        inline void reserve(uint32_t n) { r.reserve(int(n)); }
        inline uint32_t peek_n(uint32_t n) { return n ? uint32_t(r.window() >> (64 - n)) : 0; }
        inline void skip(uint32_t n) { r.skip(int(n)); }
    };
    template <> struct Coders<rlgrf::PageBuf> { using Enc = FastEnc; using RawW = FastRawW; };
    template <> struct Coders<FastIn> { using Dec = FastDec; using RawR = FastRawR; };
#if defined(RLGR_POOL3)
    // One page pool per channel. The encoder is then free to traverse the image in any order
    // (row-split, one encoder state live) while the decoder still interleaves the three chains
    // and keeps three independent table lookups in flight. Costs one extra word in the file.
    struct Pool3 { rlgrf::PageBuf* p[3]; };
    struct In3   { FastIn* p[3]; };
    template <> struct Coders<Pool3> { using Enc = FastEnc; using RawW = FastRawW; };
    template <> struct Coders<In3>   { using Dec = FastDec; using RawR = FastRawR; };
#endif
#elif LLRICE_ENGINE == 5
    struct BytesEnc : rlgrb::Encoder { explicit BytesEnc(rlgrb::Buf& b) : rlgrb::Encoder(&b) {} };
    struct BytesDec : rlgrb::Decoder { explicit BytesDec(rlgrb::In& i) : rlgrb::Decoder(&i) {} };
    struct BytesRawW {
        rlgrb::Writer w; explicit BytesRawW(rlgrb::Buf& b) : w(&b) {}
        inline void reserve(int n) { w.reserve(n); }
        inline void put_bits(uint32_t n, uint32_t v) { if (n) w.put(int(n), v); }
        inline void flush() { w.flush(); }
    };
    struct BytesRawR {
        rlgrb::Reader r; explicit BytesRawR(rlgrb::In& i) : r(&i) {}
        inline void reserve(uint32_t n) { r.reserve(int(n)); }
        inline uint32_t peek_n(uint32_t n) { if (!n) return 0; if (r.valid < int(n)) r.refill(); return uint32_t(r.win >> (64 - n)); }
        inline void skip(uint32_t n) { r.skip(int(n)); }
    };
    template <> struct Coders<rlgrb::Buf> { using Enc = BytesEnc; using RawW = BytesRawW; };
    template <> struct Coders<rlgrb::In> { using Dec = BytesDec; using RawR = BytesRawR; };
#elif LLRICE_ENGINE == 3 || LLRICE_ENGINE == 4
#if LLRICE_ENGINE == 3
    namespace lane_ns = rlgrl;
#else
    namespace lane_ns = rlgrl2;
#endif
    struct LaneEnc : lane_ns::Encoder { explicit LaneEnc(rlgrl::Lane& l) : lane_ns::Encoder(&l) {} };
    struct LaneDec : lane_ns::Decoder { explicit LaneDec(rlgrl::LaneR& r) : lane_ns::Decoder(&r) {} };
    struct LaneRawW {
        rlgrl::Lane& l; explicit LaneRawW(rlgrl::Lane& l_) : l(l_) {}
        inline void reserve(int) {}
        inline void put_bits(uint32_t n, uint32_t v) { if (n) l.put(int(n), v); }
        inline void flush() {}
    };
    struct LaneRawR {
        rlgrl::LaneR& r; explicit LaneRawR(rlgrl::LaneR& r_) : r(r_) {}
        inline void reserve(uint32_t) {}
        inline uint32_t peek_n(uint32_t n) { return n ? uint32_t(r.window() >> (64 - n)) : 0; }
        inline void skip(uint32_t n) { r.skip(int(n)); }
    };
    template <> struct Coders<rlgrl::Lane> { using Enc = LaneEnc; using RawW = LaneRawW; };
    template <> struct Coders<rlgrl::LaneR> { using Dec = LaneDec; using RawR = LaneRawR; };
#endif

    template <typename T>
    inline T median(T a, T b, T c)
    {
        //return a + b + c - std::max({a, b, c}) - std::min({a, b, c});
        return std::clamp(c, std::min(a, b), std::max(a, b));
    }

    struct Header
    {
        uint32_t hdr_crc;
        uint32_t width;         // same for all channels
        uint32_t height;        // same for all channels
        uint16_t flags;          // 15: is rgb (first 3 channels is red, green, blue channels
        uint8_t  channel_nb;    // r,g,b,n1,n2,n3,n4...
        uint8_t  channels_depth; // 8-16 supported: 8-15 -> int16_t arithmetic coding, 16 -> int32_t arithmetic coding, >16 is not tested
        void protect() {
            hdr_crc = crc32_u32(signature, width);
            hdr_crc = crc32_u32(hdr_crc, height);
            hdr_crc = crc32_u16(hdr_crc, flags);
            hdr_crc = crc32_u8(hdr_crc, channel_nb);
            hdr_crc = crc32_u8(hdr_crc, channels_depth);
        }
        bool check()  {
            auto crc = crc32_u32(signature, width);
            crc = crc32_u32(crc, height);
            crc = crc32_u16(crc, flags);
            crc = crc32_u8(crc, channel_nb);
            crc = crc32_u8(crc, channels_depth);
            return crc == hdr_crc;
        }
    };


    template <typename T>
    inline auto predict(T l, T t, T tl) {
        return median<T>(l, l + t - tl, t);
    }


#ifdef __clang__
typedef int16_t v4i16 __attribute__((ext_vector_type(4)));
typedef int16_t v4i32 __attribute__((ext_vector_type(4)));
#else
typedef int16_t v4i16 __attribute__ ((vector_size (4 * sizeof(int16_t))));
typedef int16_t v4i32 __attribute__ ((vector_size (4 * sizeof(int32_t))));
#endif

template <typename Load>
void LOCO_I(Load && load, size_t width, size_t height) {
    using Int = std::remove_cv_t<std::remove_reference_t<decltype(load())>>;   // C++17 spelling of remove_cvref_t
    auto up_line = std::vector(width,Int{});
    for (int h=0;h<height; ++h) {
        auto left = up_line[0];
        auto ptop = up_line[0];
        for (int x=0;x<width;  ++x) {
            Int top  = up_line[x];
            auto cur = up_line[x] = Load( median(top,left,Int( top + left - ptop)), x,h );
            ptop =  top;
            left = up_line[x] = cur;
        }
    }
}



    template <typename T, typename Pool, typename RGBLoader>
    void rlgr_encode(Pool& out_pool, uint32_t width, uint32_t height, RGBLoader && rgbloader)
    {
        uint32_t stride = width;
        std::vector<T> lines_storage(width * 3 * 2);
        std::array<std::array<T*,3>,2> lines
            {
                std::array{lines_storage.data(),lines_storage.data()+stride,lines_storage.data()+stride*2},
                std::array{lines_storage.data()+stride*3,lines_storage.data()+stride*4,lines_storage.data()+stride*5},

            };


        using Encoder = typename Coders<Pool>::Enc;

#if defined(RLGR_POOL3)
        Encoder g_encoder(*out_pool.p[0]);
#else
        Encoder g_encoder(out_pool);
#endif
        #if CODER == CODER_G_RB  ||  CODER == CODER_G_RB_SUM
        Encoder rb_encoder(out_pool);
        #endif
        #if CODER == CODER_G_RB_SUM
            typename Coders<Pool>::RawW rem_encoder(out_pool);
        #endif
        #if CODER == CODER_G_R_B
#if defined(RLGR_POOL3)
            Encoder r_encoder(*out_pool.p[1]);
            Encoder b_encoder(*out_pool.p[2]);
#else
            Encoder r_encoder(out_pool);
            Encoder b_encoder(out_pool);
#endif
        #endif


        // uint16_t Rshift = std::max(0, hdr.channels_depth - 8);
        // uint16_t Rmask = (1 << Rshift) - 1;

        // auto quant3 = [](auto v) {
        //     return -(v < -4) + (v > 4);
        // };


        uint32_t rem[3][16];

#if defined(RLGR_ROW_SPLIT) && CODER == CODER_G_R_B
        std::vector<uint32_t> vbuf(width);   // row-split: one channel of residuals, L1-resident
#endif
        for (uint32_t h = 0; h < height; ++h) {
            auto& curr_line = lines[h & 1];
            auto& prev_line = lines[(h-1) & 1];
#if defined(RLGR_ROW_SPLIT) && CODER == CODER_G_R_B
            // ---- row-split: whole row first, then one channel at a time ----
            // Only one encoder state is live in the token loop, so it can stay in registers.
            // The value order inside each channel is unchanged; what changes is the order the
            // three encoders take pages from the shared pool, so the decoder must split the same way.
            for (uint32_t i = 0; i < width; ++i)
                std::tie(curr_line[0][i],curr_line[1][i],curr_line[2][i]) = rgbloader();
            for (uint32_t i = 0; i < width; ++i) {
                curr_line[0][i] -= curr_line[1][i];
                curr_line[2][i] -= curr_line[1][i];
                curr_line[1][i] += (curr_line[0][i] + curr_line[2][i] + LLRICE_RCT_RND) >> 2;
                if (LLRICE_RCT_FNBLI) curr_line[2][i] -= curr_line[0][i] >> 2;
            }
            #define RLGR_ROW_ENC(C, ENC)                                                          \
            {                                                                                     \
                const T* cl = curr_line[C]; const T* pl = prev_line[C];                            \
                if (h == 0) {                                                                     \
                    vbuf[0] = Rice::to_unsigned(cl[0]);                                           \
                    for (uint32_t i = 1; i < width; ++i)                                          \
                        vbuf[i] = Rice::to_unsigned(cl[i] - cl[i-1]);                             \
                } else {                                                                          \
                    const uint32_t v16 = width < 16u ? width : 16u;                               \
                    for (uint32_t i = 0; i < v16; ++i)                                            \
                        vbuf[i] = Rice::to_unsigned(cl[i] - pl[i]);                               \
                    for (uint32_t i = 16; i < width; ++i)                                         \
                        vbuf[i] = Rice::to_unsigned(cl[i] - predict(cl[i-1], pl[i], pl[i-1]));    \
                }                                                                                 \
                for (uint32_t i = 0; i < width; ++i) ENC.put(vbuf[i]);                            \
            }
            RLGR_ROW_ENC(1, g_encoder)
            RLGR_ROW_ENC(0, r_encoder)
            RLGR_ROW_ENC(2, b_encoder)
            #undef RLGR_ROW_ENC
#else
            auto width16 =width & ~0xfu;
            auto widthR  =width & 0xfu;
            for (uint32_t w = 0; w < width16; w+=16) {

                    for (int i = 0; i <16; ++i) {
                        std::tie(curr_line[0][i+w],curr_line[1][i+w],curr_line[2][i+w]) = rgbloader();
                    }
                    for (int i = 0; i <16; ++i) {
                        curr_line[0][i+w] -= curr_line[1][i+w];
                        curr_line[2][i+w] -= curr_line[1][i+w];
                        curr_line[1][i+w] += (curr_line[0][i+w] + curr_line[2][i+w] + LLRICE_RCT_RND) >> 2;
                        if (LLRICE_RCT_FNBLI) curr_line[2][i+w] -= curr_line[0][i+w] >> 2;
                    }

                    if (likely(w>0 && h>0)) {
                        for (int i = 0; i <16; ++i) {
                            //predict(T l, T t, T tl) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned(curr_line[c][w+i] - predict( curr_line[c][w+i-1], prev_line[c][w+i], prev_line[c][w+i-1] ));
                            }
                        }
                    } else
                    if (h==0) {
                        for (int i = 0; i <16; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned( curr_line[c][w+i] - (w+i ? curr_line[c][w+i-1] : 0 ) );
                            }
                        }
                    } else {
                        for (int i = 0; i <16; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned( curr_line[c][w+i] - prev_line[c][w+i] );
                            }
                        }
                    }
                    for (int i = 0; i < 16; ++i) {
                        auto& r_rem = rem[0][i];
                        auto& g_rem = rem[1][i];
                        auto& b_rem = rem[2][i];

                        g_encoder.put( g_rem ); //g is luma
                        #if CODER == CODER_GRB
                        g_encoder.put( r_rem );
                        g_encoder.put( b_rem );
                        #endif
                        #if CODER == CODER_G_RB
                            rb_encoder.put( r_rem );
                            rb_encoder.put( b_rem );
                        #endif

                        #if CODER == CODER_G_RB_SUM
                            auto rb = r_rem + b_rem;    //rb is chroma
                            rb_encoder.put( rb );
                            if (unlikely(rb != 0)) {
                            #if TRUNC_SPLIT
                                const uint32_t tk = std20::bit_width( rb + 1 ) - 1, tu = (2u << tk) - (rb + 1);
                                rem_encoder.reserve( tk + 1 );
                                if ( r_rem < tu ) { if (tk) rem_encoder.put_bits( tk, r_rem ); }
                                else rem_encoder.put_bits( tk + 1, r_rem + tu );
                            #else
                                auto rb_width = std20::bit_width( rb );
                                rem_encoder.reserve( rb_width );
                                rem_encoder.put_bits( rb_width, r_rem );
                            #endif
                            }
                        #endif

                        #if CODER == CODER_G_R_B
                        r_encoder.put( r_rem );
                        b_encoder.put( b_rem );
                        #endif
                    }
            }
            if ( widthR > 0) {
                auto w = width16;

                    for (int i = 0; i < widthR; ++i) {
                        std::tie(curr_line[0][i+w],curr_line[1][i+w],curr_line[2][i+w]) = rgbloader();
                    }
                    for (int i = 0; i < widthR; ++i) {
                        curr_line[0][i+w] -= curr_line[1][i+w];
                        curr_line[2][i+w] -= curr_line[1][i+w];
                        curr_line[1][i+w] += (curr_line[0][i+w] + curr_line[2][i+w] + LLRICE_RCT_RND) >> 2;
                        if (LLRICE_RCT_FNBLI) curr_line[2][i+w] -= curr_line[0][i+w] >> 2;
                    }

                    if (likely(w>0 && h>0)) {
                        for (int i = 0; i < widthR; ++i) {
                            //predict(T l, T t, T tl) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned(curr_line[c][w+i] - predict( curr_line[c][w+i-1], prev_line[c][w+i], prev_line[c][w+i-1] ));
                            }
                        }
                    } else
                    if (h==0) {
                        for (int i = 0; i < widthR; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned( curr_line[c][w+i] - (w+i ? curr_line[c][w+i-1] : 0 ) );
                            }
                        }
                    } else {
                        for (int i = 0; i < widthR; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               rem[c][i] = Rice::to_unsigned( curr_line[c][w+i] - prev_line[c][w+i] );
                            }
                        }
                    }
                    for (int i = 0; i < widthR; ++i) {
                        auto& r_rem = rem[0][i];
                        auto& g_rem = rem[1][i];
                        auto& b_rem = rem[2][i];

                        g_encoder.put( g_rem ); //g is luma
                        #if CODER == CODER_GRB
                        g_encoder.put( r_rem );
                        g_encoder.put( b_rem );
                        #endif
                        #if CODER == CODER_G_RB
                            rb_encoder.put( r_rem );
                            rb_encoder.put( b_rem );
                        #endif

                        #if CODER == CODER_G_RB_SUM
                            auto rb = r_rem + b_rem;    //rb is chroma
                            rb_encoder.put( rb );
                            if (unlikely(rb != 0)) {
                            #if TRUNC_SPLIT
                                const uint32_t tk = std20::bit_width( rb + 1 ) - 1, tu = (2u << tk) - (rb + 1);
                                rem_encoder.reserve( tk + 1 );
                                if ( r_rem < tu ) { if (tk) rem_encoder.put_bits( tk, r_rem ); }
                                else rem_encoder.put_bits( tk + 1, r_rem + tu );
                            #else
                                auto rb_width = std20::bit_width( rb );
                                rem_encoder.reserve( rb_width );
                                rem_encoder.put_bits( rb_width, r_rem );
                            #endif
                            }
                        #endif

                        #if CODER == CODER_G_R_B
                        r_encoder.put( r_rem );
                        b_encoder.put( b_rem );
                        #endif
                    }
            }
#endif

        #ifdef USE_NEW_LINE
            g_encoder.new_line();
            #if CODER == CODER_G_RB || CODER == CODER_G_RB_SUM
                rb_encoder.new_line();
            #endif
            #if CODER == CODER_G_R_B
                r_encoder.new_line();
                b_encoder.new_line();
            #endif
        #endif

        }
        g_encoder.flush();
        #if CODER == CODER_G_RB || CODER == CODER_G_RB_SUM
            rb_encoder.flush();
        #endif
        #if CODER == CODER_G_RB_SUM
            rem_encoder.flush();
        #endif
        #if CODER == CODER_G_R_B
            r_encoder.flush();
            b_encoder.flush();
        #endif
    }


    template <typename UT, typename T, typename Pool, typename Store>
    inline void rlgr_decode(Pool& pool, Store && store,  uint32_t width, uint32_t height)
    {
        std::array<std::array<std::vector<T>,3>,2> lines
            {
                std::array{std::vector<T>(width),std::vector<T>(width),std::vector<T>(width)},
                std::array{std::vector<T>(width),std::vector<T>(width),std::vector<T>(width)}
            };
        using Decoder = typename Coders<Pool>::Dec;
#if defined(RLGR_POOL3)
        Decoder g_decoder(*pool.p[0]);
#else
        Decoder g_decoder(pool);
#endif
        #if  CODER == CODER_G_RB || CODER == CODER_G_RB_SUM
        Decoder rb_decoder(pool);
        #endif
        #if CODER == CODER_G_RB_SUM
            typename Coders<Pool>::RawR rem_bs(pool);
        #endif

        #if CODER == CODER_G_R_B
#if defined(RLGR_POOL3)
            Decoder r_decoder(*pool.p[1]);
            Decoder b_decoder(*pool.p[2]);
#else
            Decoder r_decoder(pool);
            Decoder b_decoder(pool);
#endif
        #endif

        uint32_t rem[3][16];

#if defined(RLGR_ROW_SPLIT) && CODER == CODER_G_R_B
        std::vector<uint32_t> vbuf(width);   // row-split: one channel of residuals, L1-resident
#endif
        for (uint32_t h = 0; h < height; ++h) {
            auto& curr_line = lines[h & 1];
            auto& prev_line = lines[(h-1) & 1];
#if defined(RLGR_ROW_SPLIT) && !defined(RLGR_POOL3) && CODER == CODER_G_R_B
            // ---- row-split: mirrors the encoder, one decoder state live at a time ----
            #define RLGR_ROW_DEC(C, DEC)                                                          \
            {                                                                                     \
                T* cl = curr_line[C].data(); const T* pl = prev_line[C].data();                    \
                if (h == 0) {                                                                     \
                    cl[0] = Rice::to_signed(DEC.get());                                           \
                    for (uint32_t i = 1; i < width; ++i)                                          \
                        cl[i] = Rice::to_signed(DEC.get()) + cl[i-1];                             \
                } else {                                                                          \
                    const uint32_t v16 = width < 16u ? width : 16u;                               \
                    for (uint32_t i = 0; i < v16; ++i)                                            \
                        cl[i] = Rice::to_signed(DEC.get()) + pl[i];                               \
                    for (uint32_t i = 16; i < width; ++i)                                         \
                        cl[i] = Rice::to_signed(DEC.get()) + predict(cl[i-1], pl[i], pl[i-1]);    \
                }                                                                                 \
            }
            RLGR_ROW_DEC(1, g_decoder)
            RLGR_ROW_DEC(0, r_decoder)
            RLGR_ROW_DEC(2, b_decoder)
            #undef RLGR_ROW_DEC
            for (uint32_t i = 0; i < width; ++i) {
                T r = curr_line[0][i], g = curr_line[1][i], b = curr_line[2][i];
                if (LLRICE_RCT_FNBLI) b += r >> 2; g -= (r + b + LLRICE_RCT_RND) >> 2; r += g; b += g;
                store(static_cast<UT>(r),static_cast<UT>(g),static_cast<UT>(b));
            }
#else
            auto width16 =width & ~0xfu;
            auto widthR  =width & 0xfu;

            for (uint32_t w = 0; w < width16; w+=16) {
                for (int i = 0; i <16; ++i) {
                auto& r = rem[0][i];
                auto& g = rem[1][i];
                auto& b = rem[2][i];

                g = g_decoder.get();
                #if CODER == CODER_G_R_B
                    r = r_decoder.get();
                    b = b_decoder.get();
                #endif
                #if CODER == CODER_GRB
                    r = g_decoder.get();
                    b = g_decoder.get();
                #endif
                #if CODER == CODER_G_RB
                    r = rb_decoder.get();
                    b = rb_decoder.get();
                #endif
                #if CODER == CODER_G_RB_SUM
                    r = 0;
                    b = 0;
                    uint32_t rb = rb_decoder.get();
                    if (unlikely(rb != 0)) {
                    #if TRUNC_SPLIT
                        const uint32_t tk = std20::bit_width(rb + 1) - 1, tu = (2u << tk) - (rb + 1);
                        rem_bs.reserve(tk + 1);
                        uint32_t r_unsigned = 0;
                        if (tk) { r_unsigned = rem_bs.peek_n(tk); rem_bs.skip(tk); }
                        if (r_unsigned >= tu) { r_unsigned = ((r_unsigned << 1) | rem_bs.peek_n(1)) - tu; rem_bs.skip(1); }
                    #else
                        auto rb_width = std20::bit_width(rb);
                        rem_bs.reserve(rb_width);
                        auto r_unsigned = rem_bs.peek_n(rb_width);
                        rem_bs.skip(rb_width);
                    #endif
                        r = r_unsigned;
                        b = rb - r_unsigned;
                    }
                #endif
                }

                    if (likely(w>0 && h>0)) {
                        for (int i = 0; i <16; ++i) {
                            //predict(T l, T t, T tl) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] =  Rice::to_signed(rem[c][i]) +  predict( curr_line[c][w+i-1], prev_line[c][w+i], prev_line[c][w+i-1]);
                            }
                        }
                    } else
                    if (h==0) {
                        for (int i = 0; i <16; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] = Rice::to_signed(rem[c][i])  + (w+i ? curr_line[c][w+i-1] : 0 );
                            }
                        }
                    } else {
                        for (int i = 0; i <16; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] = Rice::to_signed(rem[c][i])  + prev_line[c][w+i];
                            }
                        }
                    }
                    for (int i = 0; i <16; ++i) {
                        T r = curr_line[0][w+i];
                        T g = curr_line[1][w+i];
                        T b = curr_line[2][w+i];
                        if (LLRICE_RCT_FNBLI) b += r >> 2;
                        g -= (r + b + LLRICE_RCT_RND) >> 2;
                        r += g;
                        b += g;
                        store(static_cast<UT>(r),static_cast<UT>(g),static_cast<UT>(b));
                    }
            }

            if ( widthR > 0) {
                auto w = width16;
                for (int i = 0; i < widthR; ++i) {
                auto& r = rem[0][i];
                auto& g = rem[1][i];
                auto& b = rem[2][i];

                g = g_decoder.get();
                #if CODER == CODER_G_R_B
                    r = r_decoder.get();
                    b = b_decoder.get();
                #endif
                #if CODER == CODER_GRB
                    r = g_decoder.get();
                    b = g_decoder.get();
                #endif
                #if CODER == CODER_G_RB
                    r = rb_decoder.get();
                    b = rb_decoder.get();
                #endif
                #if CODER == CODER_G_RB_SUM
                    r = 0;
                    b = 0;
                    uint32_t rb = rb_decoder.get();
                    if (unlikely(rb != 0)) {
                    #if TRUNC_SPLIT
                        const uint32_t tk = std20::bit_width(rb + 1) - 1, tu = (2u << tk) - (rb + 1);
                        rem_bs.reserve(tk + 1);
                        uint32_t r_unsigned = 0;
                        if (tk) { r_unsigned = rem_bs.peek_n(tk); rem_bs.skip(tk); }
                        if (r_unsigned >= tu) { r_unsigned = ((r_unsigned << 1) | rem_bs.peek_n(1)) - tu; rem_bs.skip(1); }
                    #else
                        auto rb_width = std20::bit_width(rb);
                        rem_bs.reserve(rb_width);
                        auto r_unsigned = rem_bs.peek_n(rb_width);
                        rem_bs.skip(rb_width);
                    #endif
                        r = r_unsigned;
                        b = rb - r_unsigned;
                    }
                #endif
                }

                    if (likely(w>0 && h>0)) {
                        for (int i = 0; i < widthR; ++i) {
                            //predict(T l, T t, T tl) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] =  Rice::to_signed(rem[c][i]) +  predict( curr_line[c][w+i-1], prev_line[c][w+i], prev_line[c][w+i-1]);
                            }
                        }
                    } else
                    if (h==0) {
                        for (int i = 0; i < widthR; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] = Rice::to_signed(rem[c][i])  + (w+i ? curr_line[c][w+i-1] : 0 );
                            }
                        }
                    } else {
                        for (int i = 0; i < widthR; ++i) {
                            for (int c = 0; c < 3; ++c) {
                               curr_line[c][w+i] = Rice::to_signed(rem[c][i])  + prev_line[c][w+i];
                            }
                        }
                    }
                    for (int i = 0; i < widthR; ++i) {
                        T r = curr_line[0][w+i];
                        T g = curr_line[1][w+i];
                        T b = curr_line[2][w+i];
                        if (LLRICE_RCT_FNBLI) b += r >> 2;
                        g -= (r + b + LLRICE_RCT_RND) >> 2;
                        r += g;
                        b += g;
                        store(static_cast<UT>(r),static_cast<UT>(g),static_cast<UT>(b));
                    }
            }
#endif

        #ifdef USE_NEW_LINE
            g_decoder.new_line();
            #if CODER == CODER_G_RB || CODER == CODER_G_RB_SUM
                rb_decoder.new_line();
            #endif

            #if CODER == CODER_G_R_B
                r_decoder.new_line();
                b_decoder.new_line();
            #endif
        #endif
        }
    }


    inline auto compress(const RawImage& img)
    {
        Header hdr{0, img.width, img.height,static_cast<uint16_t>(1 << 15), img.channel_nb, img.bits_per_channel };
#if defined(RLGR_POOL3)
        hdr.flags |= uint16_t(1u << 14);   // per-channel page pools
#endif
        hdr.protect();
#if LLRICE_ENGINE == 9
        if (img.channel_type != RawImage::ChannelType::uint8 || img.channel_nb != 3) throw std::runtime_error("ENGINE 9: 8-bit RGB only");
        {
            const auto body = rlh9::encode(img.as<uint8_t>(), int(hdr.width), int(hdr.height));
            std::vector<uint64_t> out(2 + (body.size() + 7) / 8, 0);
            std::memcpy(out.data(), &hdr, sizeof(Header));
            if (!body.empty()) std::memcpy(out.data() + 2, body.data(), body.size());
            return out;
        }
#endif
#if defined(RLGR_POOL3)
        rlgrf::PageBuf pg, pr, pb;
        const size_t guess = size_t(hdr.width) * hdr.height / 24 + 1024;
        pg.w.resize(guess); pr.w.resize(guess); pb.w.resize(guess);
        Pool3 out_pool{ {&pg, &pr, &pb} };

#elif LLRICE_ENGINE == 5
        rlgrb::Buf out_pool;                        // the header takes the first 16 bytes
        out_pool.b.resize(size_t(hdr.width) * hdr.height + 1024);   // grows on demand
        out_pool.head = sizeof(Header);
        std::memcpy(out_pool.b.data(), &hdr, sizeof(Header));
#elif LLRICE_ENGINE == 3 || LLRICE_ENGINE == 4
        rlgrl::Lane out_pool;                       // the header takes the first 128 bits of the lane
        out_pool.w.resize(size_t(hdr.width) * hdr.height / 8 + 1024);   // grows on demand
        out_pool.head = 128;
        std::memcpy(out_pool.w.data(), &hdr, sizeof(Header));
#elif LLRICE_ENGINE
        rlgrf::PageBuf out_pool;
        out_pool.w.resize(size_t(hdr.width) * hdr.height / 8 + 1024);   // grows on demand
        out_pool.take(); //для заголовка
        out_pool.take();
        std::memcpy(out_pool.w.data(), &hdr, sizeof(Header));
#else
        PagePool out_pool;
        out_pool.reserve(hdr.width*hdr.height);
        out_pool.acquire_page(); //для заголовка
        out_pool.acquire_page();
        std::memcpy(&(out_pool[0]), &hdr, sizeof(Header));
#endif

        if ( img.channel_type == RawImage::ChannelType::uint16 ) {
            auto pixels = img.as<uint16_t>();
        //auto rlgr_encode(PagePool& out_pool, uint32_t width, uint32_t height, RGBLoader && rgbloader, FoldFunc && fold)

            rlgr_encode<int32_t>(out_pool, hdr.width, hdr.height,[&]()
            {
                pixels += 3;
                return std::make_tuple( pixels[-3], pixels[-2], pixels[-1]);
            });
        } else
        if ( img.channel_type == RawImage::ChannelType::uint8 ) {
            auto pixels = img.as<uint8_t>();
            rlgr_encode<int16_t>(out_pool, hdr.width, hdr.height,[&]()
            {
                pixels += 3;
                return std::make_tuple( pixels[-3], pixels[-2], pixels[-1]);
            });
        } else
        throw std::runtime_error("Unsupported channel type");
#if defined(RLGR_POOL3)
        std::vector<uint64_t> out(3, 0);
        std::memcpy(out.data(), &hdr, sizeof(Header));
        out[2] = uint64_t(pg.top) | (uint64_t(pr.top) << 32);
        out.insert(out.end(), pg.w.begin(), pg.w.begin() + pg.top);
        out.insert(out.end(), pr.w.begin(), pr.w.begin() + pr.top);
        out.insert(out.end(), pb.w.begin(), pb.w.begin() + pb.top);
        return out;
#elif LLRICE_ENGINE == 5
        std::vector<uint64_t> out((out_pool.head + 7) / 8, 0);
        std::memcpy(out.data(), out_pool.b.data(), out_pool.head);
        return out;
#elif LLRICE_ENGINE == 3 || LLRICE_ENGINE == 4
        out_pool.w.resize(out_pool.words());
        return std::move(out_pool.w);
#elif LLRICE_ENGINE
        out_pool.w.resize(out_pool.top);
        return std::move(out_pool.w);
#else
        return out_pool.move();
#endif
    }


    inline bool isLittleEndian() {
        uint16_t num = 0x0001; // 1
        return *(uint8_t*)&num == 0x01;
    }

    inline RawImage decompress(std::vector<uint64_t>& poolvec)
    {
        size_t rgb_pos = 0;
        Header hdr;
        RawImage img;

        if (poolvec.size() < 2) throw std::runtime_error("truncated file");
#if LLRICE_ENGINE == 5
        rlgrb::In pool{reinterpret_cast<const uint8_t*>(poolvec.data()), poolvec.size() * 8, sizeof(Header)};
#elif LLRICE_ENGINE == 3 || LLRICE_ENGINE == 4
        const size_t nwords = poolvec.size();
        poolvec.push_back(0); poolvec.push_back(0);  // the reader peeks two words ahead
        rlgrl::LaneR pool; pool.w = poolvec.data(); pool.nw = nwords + 2; pool.head = 128;   //пропускаем заголовок
#elif defined(RLGR_POOL3)
        // the three cursors are built once the header has been read
#elif LLRICE_ENGINE
        FastIn pool{poolvec.data(), 2, poolvec.size()};   //пропускаем заголовок
#else
        PagePool pool(std::move(poolvec));
        pool.get_next_read_page(); //пропускаем заголовок
        pool.get_next_read_page();
#endif

#if LLRICE_ENGINE
        std::memcpy(&hdr, poolvec.data(), sizeof(Header));
#else
        std::memcpy(&hdr, &(pool[0]), sizeof(Header));
#endif
        if (!hdr.check()) {
            throw std::runtime_error("Header CRC check failed");
        }
        img.width = hdr.width;
        img.height = hdr.height;
        img.channel_nb = 3;
        img.bits_per_channel = hdr.channels_depth;
        //RawImage::ChannelType::uint16be  pnm freadly
        img.channel_type = hdr.channels_depth <= 8 ? RawImage::ChannelType::uint8 : RawImage::ChannelType::uint16be;
        img.allocate();
#if LLRICE_ENGINE == 9
        if (img.channel_type != RawImage::ChannelType::uint8) throw std::runtime_error("ENGINE 9: 8-bit RGB only");
        {
            const size_t nbody = (poolvec.size() - 2) * 8;
            poolvec.resize(poolvec.size() + (rlh9::decode_pad(int(img.width)) + 15) / 8, 0);   // zero tail for the bit peeks
            rlh9::decode(reinterpret_cast<const uint8_t*>(poolvec.data() + 2), nbody, img.as<uint8_t>(), int(img.width), int(img.height));
            return img;
        }
#endif
#if defined(RLGR_POOL3)
        if (!(hdr.flags & (1u << 14))) throw std::runtime_error("not a per-channel-pool stream");
        if (poolvec.size() < 3) throw std::runtime_error("truncated file");
        const size_t ng = uint32_t(poolvec[2]), nr = uint32_t(poolvec[2] >> 32);
        if (3 + ng + nr > poolvec.size()) throw std::runtime_error("truncated page table");
        FastIn ig{poolvec.data() + 3,          0, ng};
        FastIn ir{poolvec.data() + 3 + ng,     0, nr};
        FastIn ib{poolvec.data() + 3 + ng + nr, 0, poolvec.size() - 3 - ng - nr};
        In3 pool{ {&ig, &ir, &ib} };
#endif
        auto width = img.width;
        auto height = img.height;
        if ( img.channel_type == RawImage::ChannelType::uint16 ) {
            auto rgb = img.as<uint16_t>();
            if (isLittleEndian()) {
                rlgr_decode<uint16_t,int32_t>(pool, [&](auto r, auto g, auto b) {
                    rgb[rgb_pos++] = bswap16(r);
                    rgb[rgb_pos++] = bswap16(g);
                    rgb[rgb_pos++] = bswap16(b);
                },  width, height);
            } else {
                rlgr_decode<uint16_t,int32_t>(pool, [&](auto r, auto g, auto b) {
                    rgb[rgb_pos++] = r;
                    rgb[rgb_pos++] = g;
                    rgb[rgb_pos++] = b;
                },  width, height);
            }
        } else
        if ( img.channel_type == RawImage::ChannelType::uint8 ) {
            auto rgb = img.as<uint8_t>();
            rlgr_decode<uint8_t,int16_t>(pool, [&](auto r, auto g, auto b) {
                rgb[rgb_pos++] = r;
                rgb[rgb_pos++] = g;
                rgb[rgb_pos++] = b;
            },  width, height);
        }else
        {
            throw std::runtime_error("Unsupported channel type");
        }
        return img;
    }

}