#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cassert>
#include "../../llrice.hpp"

void verify_roundtrip(const RawImage& img, const char* label) {
    auto words = llrice::compress(img);
    // Check signature rev == '6'
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words.data());
    static_assert(llrice::rev == '6');
    auto dec = llrice::decompress(words);
    assert(dec.width == img.width && dec.height == img.height);
    assert(dec.bits_per_channel == img.bits_per_channel);
    if (img.bits_per_channel <= 8) {
        size_t sz = size_t(img.width) * img.height * 3;
        assert(std::memcmp(img.as<uint8_t>(), dec.as<uint8_t>(), sz) == 0);
    } else {
        size_t np = size_t(img.width) * img.height * 3;
        const uint16_t* src = img.as<uint16_t>();
        const uint16_t* dst = dec.as<uint16_t>();
        for (size_t i = 0; i < np; ++i) {
            uint16_t d = llrice::isLittleEndian() ? bswap16(dst[i]) : dst[i];
            assert(src[i] == d);
        }
    }
}

void test_8bit(uint32_t w, uint32_t h, int mode, std::mt19937& rng) {
    RawImage img;
    img.width = w; img.height = h; img.channel_nb = 3; img.bits_per_channel = 8;
    img.channel_type = RawImage::ChannelType::uint8;
    img.allocate();
    uint8_t* p = img.as<uint8_t>();
    size_t n = size_t(w) * h * 3;
    if (mode == 0) {
        std::memset(p, 0, n);
    } else if (mode == 1) {
        std::memset(p, 255, n);
    } else if (mode == 2) {
        for (size_t i = 0; i < n; ++i) p[i] = uint8_t(rng() & 255);
    } else if (mode == 3) {
        // Sparse impulses separated by long zero runs
        std::memset(p, 128, n);
        for (size_t i = 0; i < n; i += (rng() % 5000 + 1)) p[i] = uint8_t(rng() & 255);
    } else if (mode == 4) {
        // Checkerboard extreme edges (maximizes RCT + MED residuals up to +-1023 -> v up to 2047)
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                bool odd = ((x + y) & 1) != 0;
                p[(y * w + x) * 3 + 0] = odd ? 255 : 0;
                p[(y * w + x) * 3 + 1] = odd ? 0 : 255;
                p[(y * w + x) * 3 + 2] = odd ? 255 : 0;
            }
        }
    }
    verify_roundtrip(img, "8bit");
}

void test_16bit(uint32_t w, uint32_t h, std::mt19937& rng) {
    size_t n = size_t(w) * h;
    std::vector<uint32_t> vals(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = rng() % 10;
        if (r < 4) vals[i] = 0;
        else if (r < 8) vals[i] = rng() % 2048;
        else vals[i] = rng() % 262144; // up to 18-bit residuals (vk up to 18)
    }
    rlgrf::PageBuf pb;
    pb.w.resize(n + 1024);
    {
        rlh::Encoder enc(&pb, 0);
        for (uint32_t v : vals) enc.put(v);
        enc.flush();
    }
    size_t top = 0;
    rlh::Decoder dec(pb.w.data(), &top, pb.top, 0);
    for (size_t i = 0; i < n; ++i) {
        assert(dec.get() == vals[i]);
    }
}

int main() {
    std::mt19937 rng(12345);
    const std::pair<uint32_t, uint32_t> dims[] = {
        {1, 1}, {1, 2}, {2, 1}, {1, 16}, {16, 1}, {1, 17}, {17, 1},
        {3, 17}, {17, 3}, {15, 15}, {16, 16}, {17, 17}, {31, 33},
        {64, 64}, {70, 1025}, {1025, 70}, {513, 513}
    };
    for (auto [w, h] : dims) {
        for (int mode = 0; mode < 5; ++mode) {
            test_8bit(w, h, mode, rng);
        }
    }
    // Test giant zero run > 262,144 pixels (e.g. 1000x600 = 600,000 pixels solid + trailing)
    test_8bit(1000, 600, 0, rng);
    test_8bit(1000, 600, 1, rng);
    test_8bit(1000, 600, 3, rng);
    // Test 16-bit images (escape path for v >= 2048 and vk >= 15)
    test_16bit(1, 1, rng);
    test_16bit(17, 19, rng);
    test_16bit(128, 128, rng);

    // Direct stream test for MAX_RUN_ENC (> 67M zeros!)
    {
        rlgrf::PageBuf pb;
        pb.w.resize(1024);
        {
            rlh::Encoder enc(&pb, 0);
            const uint32_t N = (1u << 26) + 500000u; // > MAX_RUN_ENC
            for (uint32_t i = 0; i < N; ++i) enc.put(0);
            enc.put(12345); // > 2048 escape after > 67M zeros
            for (uint32_t i = 0; i < 300000u; ++i) enc.put(0); // trailing run > 262,144 at flush()
            enc.flush();
        }
        size_t top = 0;
        rlh::Decoder dec(pb.w.data(), &top, pb.top, 0);
        const uint32_t N = (1u << 26) + 500000u;
        for (uint32_t i = 0; i < N; ++i) assert(dec.get() == 0);
        assert(dec.get() == 12345u);
        for (uint32_t i = 0; i < 300000u; ++i) assert(dec.get() == 0);
    }
    printf("All ASan+UBSan edge, 16-bit, and >67M run tests PASSED (CODER=%d)!\n", CODER);
    return 0;
}
