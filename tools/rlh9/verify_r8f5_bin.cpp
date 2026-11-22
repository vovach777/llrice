// Standalone decoder verification binary (compile separately against old and new rlh9_r8f5.hpp!)
// Usage:
//   ./dec_ref gen <out_dir> <ppm_dir1> [ppm_dir2 ...]   -> encodes edge cases + PPMs to out_dir, decodes, prints hashes
//   ./dec_new run <in_dir>                              -> decodes all .r8f4 in in_dir, prints hashes (diff against ref!)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <algorithm>
#include "../../image.hpp"
#ifdef USE_ORIG_RLH9
#include "../../rlh9_1c440de_orig.hpp"
#else
#include "../../rlh9_r8f5.hpp"
#endif

// 128-bit collision-free checksum (two independent 64-bit hashes: FNV-1a + xx-style mix)
static void hash128(const uint8_t* p, size_t n, uint64_t& h1, uint64_t& h2) {
    h1 = 14695981039346656037ULL;
    h2 = 0x9e3779b97f4a7c15ULL ^ n;
    for (size_t i = 0; i < n; ++i) {
        h1 ^= p[i]; h1 *= 1099511628211ULL;
        h2 = (h2 ^ p[i]) * 0xbf58476d1ce4e5b9ULL;
        h2 ^= h2 >> 31;
    }
}

static void save_blob(const std::string& path, int W, int H, const std::vector<uint8_t>& enc) {
    FILE* f = std::fopen(path.c_str(), "wb");
    uint32_t hdr[3] = {uint32_t(W), uint32_t(H), uint32_t(enc.size())};
    std::fwrite(hdr, 4, 3, f);
    if (!enc.empty()) std::fwrite(enc.data(), 1, enc.size(), f);
    std::fclose(f);
}

static void gen_edge(int W, int H, int mode, std::mt19937& rng, std::vector<uint8_t>& rgb) {
    rgb.resize(size_t(W) * H * 3);
    size_t n = rgb.size();
    if (mode == 0) std::memset(rgb.data(), 0, n);
    else if (mode == 1) std::memset(rgb.data(), 255, n);
    else if (mode == 2) { for (size_t i = 0; i < n; ++i) rgb[i] = uint8_t(rng() & 255); }
    else if (mode == 3) {
        std::memset(rgb.data(), 128, n);
        for (int y = 0; y < H; ++y) {
            if (W >= 2) {
                for (int c = 0; c < 3; ++c) rgb[(size_t(y) * W + (W - 1)) * 3 + c] = uint8_t(rng() & 255);
            }
            for (int x = int(rng() % 5000); x < W; x += int(rng() % 5000 + 1))
                for (int c = 0; c < 3; ++c) rgb[(size_t(y) * W + x) * 3 + c] = uint8_t(rng() & 255);
        }
    } else {
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                bool odd = ((x + y) & 1) != 0;
                rgb[(size_t(y) * W + x) * 3 + 0] = odd ? 255 : 0;
                rgb[(size_t(y) * W + x) * 3 + 1] = odd ? 0 : 255;
                rgb[(size_t(y) * W + x) * 3 + 2] = odd ? 255 : 0;
            }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::string cmd = argv[1], dir = argv[2];
    if (cmd == "gen") {
        std::filesystem::create_directories(dir);
        std::mt19937 rng(42);
        const std::pair<int, int> dims[] = {
            {1,1},{1,2},{2,1},{2,2},{1,15},{15,1},{1,16},{16,1},{1,17},{17,1},
            {2,16},{16,2},{2,17},{17,2},{3,17},{17,3},{7,9},{8,8},{9,7},
            {15,15},{15,16},{16,15},{16,16},{16,17},{17,16},{17,17},
            {31,33},{32,32},{33,31},{64,64},{513,33},
            {8192,16},{8200,16},{16382,16},{16383,1},{16383,16},{16383,17}
        };
        int idx = 0;
        std::vector<uint8_t> rgb;
        for (auto [w, h] : dims) {
            for (int mode = 0; mode < 5; ++mode) {
                gen_edge(w, h, mode, rng, rgb);
                auto enc = rlh9::encode(rgb.data(), w, h);
                char name[64]; std::snprintf(name, sizeof name, "%04d_edge_%dx%d_m%d.r8f4", idx++, w, h, mode);
                save_blob(dir + "/" + name, w, h, enc);
            }
        }
        for (int a = 3; a < argc; ++a) {
            std::vector<std::string> files;
            for (auto& e : std::filesystem::directory_iterator(argv[a]))
                if (e.path().extension() == ".ppm") files.push_back(e.path().string());
            std::sort(files.begin(), files.end());
            for (auto& f : files) {
                RawImage img; img.load(f); img.le();
                auto enc = rlh9::encode(img.as<uint8_t>(), int(img.width), int(img.height));
                char name[128]; std::snprintf(name, sizeof name, "%04d_ppm_%dx%d.r8f4", idx++, int(img.width), int(img.height));
                save_blob(dir + "/" + name, int(img.width), int(img.height), enc);
            }
        }
        std::fprintf(stderr, "Generated %d encoded .r8f4 blobs in %s\n", idx, dir.c_str());
        return 0;
    }
    if (cmd == "run") {
        std::vector<std::string> files;
        for (auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".r8f4") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        for (auto& path : files) {
            FILE* f = std::fopen(path.c_str(), "rb");
            uint32_t hdr[3];
            if (std::fread(hdr, 4, 3, f) != 3) { std::fclose(f); return 3; }
            int W = int(hdr[0]), H = int(hdr[1]); size_t n = hdr[2];
            std::vector<uint8_t> buf(n + rlh9::decode_pad(W) + 16, 0);
            if (n && std::fread(buf.data(), 1, n, f) != n) { std::fclose(f); return 4; }
            std::fclose(f);
            std::vector<uint8_t> rgb(size_t(W) * H * 3, 0);
            rlh9::decode(buf.data(), n, rgb.data(), W, H);
            uint64_t h1 = 0, h2 = 0;
            hash128(rgb.data(), rgb.size(), h1, h2);
            std::printf("%s %dx%d %016llx%016llx\n",
                        std::filesystem::path(path).filename().c_str(), W, H,
                        (unsigned long long)h1, (unsigned long long)h2);
        }
        return 0;
    }
    return 2;
}
