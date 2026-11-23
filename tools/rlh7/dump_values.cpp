// dump_values.cpp - dump the exact per-channel value streams that the llrice front-end feeds to the
// RLGR/RLH engines.  No front-end logic is duplicated: we specialise llrice::Coders<> on a sink pool
// whose "encoder" simply records put(v).  Runs of zeroes are collapsed into (rl, v) tokens here.
//
//   dump_values a.ppm|dir ... -o outdir
//
// Output: one .rlv file per image, little endian:
//   magic 'RLV1' (u32), width (u32), height (u32), nch (u32)
//   per channel: ntok (u64), then ntok records of { u32 rl; u16 v; u16 pad; }
// A token is (rl zero values, then the non-zero value v).  The last token of a channel may have v == 0
// (trailing zero run at flush).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "llrice.hpp"

struct SinkPool {
    std::vector<uint32_t> val[3];
    int opened = 0;
};

struct SinkEnc {
    SinkPool* p;
    int ch;
    explicit SinkEnc(SinkPool& d) : p(&d), ch(d.opened++) {}
    __attribute__((always_inline)) inline void put(uint32_t v) { p->val[ch].push_back(v); }
    inline void flush() {}
};

namespace llrice {
    template <> struct Coders<SinkPool> { using Enc = SinkEnc; };
}

struct Tok { uint32_t rl; uint16_t v; uint16_t pad; };

int main(int argc, char** argv) {
    std::string outdir = "dumps";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) { outdir = argv[++i]; continue; }
        std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(p))
                if (e.path().extension() == ".ppm" || e.path().extension() == ".pgm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end());
            files.insert(files.end(), v.begin(), v.end());
        } else files.push_back(p.string());
    }
    std::filesystem::create_directories(outdir);
    size_t tot_tok = 0, tot_px = 0;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        if (img.channel_type != RawImage::ChannelType::uint8) { fprintf(stderr, "skip (not 8-bit): %s\n", f.c_str()); continue; }
        SinkPool sink;
        auto pixels = img.as<uint8_t>();
        llrice::rlgr_encode<int16_t>(sink, img.width, img.height, [&]() {
            pixels += 3;
            return std::make_tuple(pixels[-3], pixels[-2], pixels[-1]);
        });
        std::filesystem::path out = std::filesystem::path(outdir) / (std::filesystem::path(f).stem().string() + ".rlv");
        FILE* g = fopen(out.c_str(), "wb");
        if (!g) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
        uint32_t hdr[4] = { 0x31564c52u, img.width, img.height, 3 };
        fwrite(hdr, 4, 4, g);
        size_t ftok = 0;
        for (int c = 0; c < 3; ++c) {
            const auto& V = sink.val[c];
            std::vector<Tok> T;
            T.reserve(V.size() / 4 + 16);
            uint32_t rl = 0;
            for (uint32_t v : V) {
                if (v == 0) { ++rl; continue; }
                T.push_back({ rl, uint16_t(v > 65535u ? 65535u : v), 0 });
                rl = 0;
            }
            if (rl) T.push_back({ rl, 0, 0 });
            uint64_t n = T.size();
            fwrite(&n, 8, 1, g);
            fwrite(T.data(), sizeof(Tok), T.size(), g);
            ftok += T.size();
        }
        fclose(g);
        tot_tok += ftok; tot_px += size_t(img.width) * img.height;
        printf("%s %ux%u tokens %zu (%.4f tok/px)\n", std::filesystem::path(f).filename().c_str(),
               img.width, img.height, ftok, double(ftok) / (double(img.width) * img.height));
    }
    printf("TOTAL %zu files %zu tokens %zu px\n", files.size(), tot_tok, tot_px);
    return 0;
}
