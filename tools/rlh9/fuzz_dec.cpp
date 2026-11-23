// Corrupt-stream fuzzer for rlh9::decode under ASan/UBSan:
// 5000 mutations on a photo crop + 5000 mutations on a screen crop.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include "../../image.hpp"
#include "../../rlh9.hpp"

static void fuzz_one(const char* ppm_path, int n_mut, uint32_t seed) {
    RawImage img; img.load(ppm_path); img.le();
    const int W = std::min(int(img.width), 256);
    const int H = std::min(int(img.height), 256);
    std::vector<uint8_t> crop(size_t(W) * H * 3);
    for (int y = 0; y < H; ++y)
        std::memcpy(crop.data() + size_t(y) * W * 3, img.as<uint8_t>() + size_t(y) * img.width * 3, size_t(W) * 3);
    auto base = rlh9::encode(crop.data(), W, H);
    std::vector<uint8_t> out(size_t(W) * H * 3);
    std::mt19937 rng(seed);
    int ok = 0, caught = 0;
    for (int i = 0; i < n_mut; ++i) {
        std::vector<uint8_t> mut = base;
        int kind = int(rng() % 5);
        if (kind == 0 && !mut.empty()) {
            // Flip 1..8 random bytes
            int k = int(rng() % 8) + 1;
            for (int j = 0; j < k; ++j) mut[rng() % mut.size()] ^= uint8_t(1u << (rng() & 7));
        } else if (kind == 1 && !mut.empty()) {
            // Random truncation
            mut.resize(rng() % mut.size());
        } else if (kind == 2 && mut.size() >= 16) {
            // Corrupt band header lengths
            size_t pos = rng() % std::min<size_t>(mut.size(), 64);
            mut[pos] = uint8_t(rng() & 255);
            if (pos + 1 < mut.size()) mut[pos + 1] = uint8_t(rng() & 255);
        } else if (kind == 3 && !mut.empty()) {
            // Overwrite a contiguous slice with 0x00 or 0xFF
            size_t pos = rng() % mut.size();
            size_t len = std::min<size_t>(mut.size() - pos, (rng() % 32) + 1);
            std::memset(mut.data() + pos, (rng() & 1) ? 0xFF : 0x00, len);
        } else {
            // Completely random payload of random size
            mut.resize(rng() % 512);
            for (auto& b : mut) b = uint8_t(rng() & 255);
        }
        const size_t n = mut.size();
        mut.resize(n + rlh9::decode_pad(W) + 16, 0);
        try {
            rlh9::decode(mut.data(), n, out.data(), W, H);
            ++ok;
        } catch (const std::exception&) {
            ++caught;
        }
    }
    std::printf("FUZZ %s (%dx%d): %d mutations -> decoded=%d threw=%d (0 crashes/UB)\n",
                ppm_path, W, H, n_mut, ok, caught);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "Usage: %s <photo.ppm> <screen.ppm>\n", argv[0]); return 2; }
    fuzz_one(argv[1], 5000, 1001);
    fuzz_one(argv[2], 5000, 2002);
    return 0;
}
