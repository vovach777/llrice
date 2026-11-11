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

//#define USE_YCoCg
//#define USE_SIMPLE_RLGR
#define CODER_GRB    0
#define CODER_G_RB   1
#define CODER_G_RB_SUM 2
#define CODER_G_R_B 3
#ifndef CODER
#define CODER CODER_G_R_B
#endif
// CODER_G_RB_SUM: r in 0..rb (rb + 1 values). TRUNC_SPLIT 1: truncated binary, k = floor(log2(rb + 1)) bits if
// r < u = 2^(k+1) - (rb + 1), else k + 1 bits of r + u (no code is wasted). 0: bit_width(rb) bits (old format).
#ifndef TRUNC_SPLIT
#define TRUNC_SPLIT 1
#endif
//#define USE_NEW_LINE
namespace llrice
{
    constexpr inline auto ext = ".llr";
    constexpr inline uint8_t rev = '1';
    constexpr inline uint32_t signature = 'l' | 'l' << 8U | 'r' << 16U | rev << 24U;

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
    using Int = std::remove_cvref_t<decltype(load())>;
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



    template <typename T, typename RGBLoader>
    auto rlgr_encode(PagePool& out_pool, uint32_t width, uint32_t height, RGBLoader && rgbloader)
    {
        uint32_t stride = width;
        std::vector<T> lines_storage(width * 3 * 2);
        std::array<std::array<T*,3>,2> lines
            {
                std::array{lines_storage.data(),lines_storage.data()+stride,lines_storage.data()+stride*2},
                std::array{lines_storage.data()+stride*3,lines_storage.data()+stride*4,lines_storage.data()+stride*5},

            };


#ifdef USE_SIMPLE_RLGR
        using Encoder = RLGR::Simple::Encoder;
#else
        using Encoder = RLGR::Encoder;
#endif

        Encoder g_encoder(out_pool);
        #if CODER == CODER_G_RB  ||  CODER == CODER_G_RB_SUM
        Encoder rb_encoder(out_pool);
        #endif
        #if CODER == CODER_G_RB_SUM
            BitStream::Writer rem_encoder(out_pool);
        #endif
        #if CODER == CODER_G_R_B
            Encoder r_encoder(out_pool);
            Encoder b_encoder(out_pool);
        #endif


        // uint16_t Rshift = std::max(0, hdr.channels_depth - 8);
        // uint16_t Rmask = (1 << Rshift) - 1;

        // auto quant3 = [](auto v) {
        //     return -(v < -4) + (v > 4);
        // };


        uint32_t rem[3][16];

        for (uint32_t h = 0; h < height; ++h) {
            auto& curr_line = lines[h & 1];
            auto& prev_line = lines[(h-1) & 1];
            auto width16 =width & ~0xfu;
            auto widthR  =width & 0xfu;
            for (uint32_t w = 0; w < width16; w+=16) {

                    for (int i = 0; i <16; ++i) {
                        std::tie(curr_line[0][i+w],curr_line[1][i+w],curr_line[2][i+w]) = rgbloader();
                    }
                    for (int i = 0; i <16; ++i) {
                        curr_line[0][i+w] -= curr_line[1][i+w];
                        curr_line[2][i+w] -= curr_line[1][i+w];
                        curr_line[1][i+w] += (curr_line[0][i+w] + curr_line[2][i+w]) >> 2;
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
                        curr_line[1][i+w] += (curr_line[0][i+w] + curr_line[2][i+w]) >> 2;
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

        return out_pool.move();
    }


    template <typename UT, typename T, typename Store>
    inline void rlgr_decode(PagePool& pool, Store && store,  uint32_t width, uint32_t height)
    {
        std::array<std::array<std::vector<T>,3>,2> lines
            {
                std::array{std::vector<T>(width),std::vector<T>(width),std::vector<T>(width)},
                std::array{std::vector<T>(width),std::vector<T>(width),std::vector<T>(width)}
            };
        using Stream = BitStream::Reader;
#ifdef USE_SIMPLE_RLGR
        using Decoder = RLGR::Simple::Decoder;
#else
        using Decoder = RLGR::Decoder;
#endif
        Decoder g_decoder(pool);
        #if  CODER == CODER_G_RB || CODER == CODER_G_RB_SUM
        Decoder rb_decoder(pool);
        #endif
        #if CODER == CODER_G_RB_SUM
            BitStream::Reader rem_bs(pool);
        #endif

        #if CODER == CODER_G_R_B
            Decoder r_decoder(pool);
            Decoder b_decoder(pool);
        #endif

        uint32_t rem[3][16];

        for (uint32_t h = 0; h < height; ++h) {
            auto& curr_line = lines[h & 1];
            auto& prev_line = lines[(h-1) & 1];
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
                        g -= (r+b) >> 2;
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
                        g -= (r+b) >> 2;
                        r += g;
                        b += g;
                        store(static_cast<UT>(r),static_cast<UT>(g),static_cast<UT>(b));
                    }
            }

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
        hdr.protect();
        PagePool out_pool;
        out_pool.reserve(hdr.width*hdr.height);
        out_pool.acquire_page(); //для заголовка
        out_pool.acquire_page();
        std::memcpy(&(out_pool[0]), &hdr, sizeof(Header));

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
        return out_pool.move();
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

        PagePool pool(std::move(poolvec));
        pool.get_next_read_page(); //пропускаем заголовок
        pool.get_next_read_page();

        std::memcpy(&hdr, &(pool[0]), sizeof(Header));
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