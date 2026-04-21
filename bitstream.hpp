#pragma once
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <iterator>
#include <cstdint>
#include <utility>
#include <cstring>
#include "polyfill.hpp"
#include "pool.hpp"
#include "ring.hpp"

namespace BitStream {

    struct Writer {
        uint64_t buf{0};
        uint32_t left{64};
        PagePool& pool;
        RingArray<size_t,4> res;
        int _left_reserved{0};

        Writer(PagePool& pool) : pool(pool) {}


        void reserve(int count) {
            //count = std::max(count,32);
            while (_left_reserved < count) {
                res.put( pool.acquire_page() );
                _left_reserved += 64;
            }
        }

    inline void put_bits(uint32_t n, uint32_t value)
    {
        if (n == 0)
            return;
        assert(n <= _left_reserved);
        assert(n <= 32);
        assert( (n == 32) || (value >> n) == 0 );
        if (n <= left) {
            buf <<= n;
            buf |= value;
            left -= n;
            _left_reserved -= n;
        } else {
            assert(left < 32);
            auto rem = n - left;
            buf <<= left;
            buf |= static_cast<uint64_t>(value) >> rem;
            assert(res.size() > 0);
            pool[ res.get() ] = (buf);
            _left_reserved -= n;
            left += 64 - n;
            buf = value; //it's ok that have extra MSB rem bits.. it's cuts on flush later
        }
    }

    inline void byte_align() {
        uint32_t shift = left & 7;
        buf <<= shift;
        left -= shift;
    }

    inline void flush() {
        if (left < 64) {
            buf <<= left;
            assert(res.size() > 0);
            pool[ res.get() ] = (buf);
            buf = 0;
            left = 64;
        }
    }

    };

    /*

    The first implementation employs a lazy reading pattern, deferring heavy operations until strictly necessary (within the refill method).
    In contrast, the second implementation aggressively attempts to maintain a ready 32-bit lookahead window on every skip invocation,
    incurring substantial computational overhead that ultimately bottlenecks the parser.

    */

#ifndef SLOW_READER
    struct alignas(32) Reader
    {
        uint64_t buf64{0}; // stores bits read from the buffer
        // uint32_t buf32{0};
        uint32_t bits_valid{0}; // number of bits left in bits field
        int32_t total_buffered{0};  // cnt32 + cnt64 + res.size()*64
        uint32_t rdy_peek{0};
        RingArray<uint64_t,4> res;
        const PagePool& pool;

        Reader(const PagePool& pool) : pool(pool) {
        }

        inline uint32_t get_total_buffered() const {
            return bits_valid + (res.size() * 64) - rdy_peek;
        }

        void reserve(uint32_t count) {
            assert(count > 0);
            while (total_buffered < count)  {
                res.put( pool.get_next_read_page() );
                total_buffered += 64;
            }
        }

        inline void skip(uint32_t n)
        {
            assert(bits_valid >= n);
            buf64 <<=  n;
            bits_valid -= n;
            total_buffered -=n;
        }
#if 1
        void refill(uint32_t count) {

            while (bits_valid < count) {
                auto take = std::min(64 - bits_valid, 64 - rdy_peek);
                if (take == 0) {
                    assert(res.size() > 0);
                    res.get();
                    rdy_peek = 0;
                    continue;
                }
                assert(res.size() > 0);
                buf64 |= (res.peek() >> bits_valid);
                bits_valid += take;
                if (take < 64) {
                    res.peek() <<= take;
                } else {
                    res.peek() = 0;
                }
                rdy_peek += take;
            }
        }
#else
        uint64_t cur = 0;
        uint32_t cur_bits = 0;

        inline void refill(uint32_t count)
        {
            while (bits_valid < count)
            {
                if (cur_bits == 0) {
                    assert(res.size() > 0);
                    cur = res.get();
                    cur_bits = 64;
                }

                uint32_t space = 64 - bits_valid;
                uint32_t take  = (cur_bits < space) ? cur_bits : space;

                // берём старшие take бит из cur
                uint64_t chunk = cur >> (64 - take);

                // кладём их сразу за уже имеющимися bits_valid битами в buf64
                buf64 |= chunk << (64 - bits_valid - take);

                cur      <<= take;
                cur_bits -= take;
                bits_valid += take;
            }
        }

#endif
        inline uint32_t peek32()
        {
            if ((bits_valid < 32)) {
                refill(32);
            }
            assert(bits_valid >= 32);
            return buf64 >> 32;


        }
        inline uint32_t peek_n(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            if (bits_valid < n) {
                refill(n);
            }
            assert(bits_valid >= n);
            return buf64 >> (64-n);
        }
    };
#else
    struct alignas(32) Reader
    {
        uint64_t buf64{0}; // stores bits read from the buffer
        uint32_t buf32{0};
        uint32_t bits_valid_64{0}; // number of bits left in bits field
        int32_t total_buffered{0};  // cnt32 + cnt64 + res.size()*64
        RingArray<uint64_t,4> res;
        const PagePool& pool;
        Reader(const PagePool& pool) : pool(pool) {}

        inline void reserve(uint32_t count) {
            auto glue_operation_if_below_32 = total_buffered;

            while (total_buffered < count)  {
               res.put( pool.get_next_read_page() );
               total_buffered += 64;
            }
            if (glue_operation_if_below_32 < 32 && res.size() > 0) {
                // if (res.size() == 0) {
                //     std::cout << "warning res.size()=0 has hole: bits_valid=" << bits_valid << " on_reserve=" << count << std::endl;
                //     return;
                // }
                //assert(res.size() > 0);
                assert(bits_valid_64 == 0);
                auto take = 32-glue_operation_if_below_32;
                //std::cout << " take = " << take << std::endl;
                assert(take > 0);

                buf64 = res.get();
                bits_valid_64 = 64-take;
                buf32 |= static_cast<uint32_t>(buf64 >> (64-take));
                buf64 <<= take;
            }

        }

        inline void skip(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            assert(total_buffered >= n);
            if (likely( n <= bits_valid_64)) {
                total_buffered -= n;
                if (unlikely(n == 32)){
                    buf32 = static_cast<uint32_t>(buf64 >> 32);
                    buf64 <<= 32;
                    bits_valid_64 -= 32;
                }else {
                    buf32 <<= n;
                    buf32 |= static_cast<uint32_t>(buf64 >> (64 - n));
                    buf64 <<= n;
                    bits_valid_64 -= n;
                }
                return;
            }

            if (bits_valid_64 > 0) {
                const uint32_t rem{bits_valid_64};
                bits_valid_64 = 0;
                buf32 <<= rem;
                buf32 |= static_cast<uint32_t>(buf64 >> (64U - rem));
                n -= rem;
                total_buffered -= rem;
            }
            assert( n != 0 );
            total_buffered -= n;

            if ( unlikely(res.size() == 0)) {
                if (unlikely(n==32))
                    buf32 = 0;
                 else
                    buf32 <<= n;
                return;
            } else {
                buf64 = res.get();
                bits_valid_64 = 64;
                if (unlikely(n == 32)){
                    buf32 = static_cast<uint32_t>(buf64 >> 32);
                    buf64 <<= 32;
                    bits_valid_64 -= 32;
                }else {
                    buf32 <<= n;
                    buf32 |= static_cast<uint32_t>(buf64 >> (64 - n));
                    buf64 <<= n;
                    bits_valid_64 -= n;
                }
            }

        }

        inline uint32_t peek32()
        {
            assert(total_buffered  >= 32);
            return buf32;
        }
        inline uint32_t peek_n(uint32_t n)
        {
            assert(n != 0);
            assert(n <= 32);
            assert(total_buffered >= n);
            return n == 32 ? buf32 : (buf32 >> (32-n));
        }
    };
#endif

}


