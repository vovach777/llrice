// bench.cpp - in-memory llrice benchmark: the image is read once, the timer covers only compress / decompress,
// best of -r runs, round trip checked. Output like palic-2 bench.  bench a.ppm|dir ... [-r N]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "llrice.hpp"
int main(int argc, char** argv) {
    int reps = 3; std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-r") && i + 1 < argc) { reps = atoi(argv[++i]); continue; }
        std::filesystem::path p(argv[i]);
        if (std::filesystem::is_directory(p)) {
            std::vector<std::string> v;
            for (auto& e : std::filesystem::directory_iterator(p)) if (e.path().extension() == ".ppm") v.push_back(e.path().string());
            std::sort(v.begin(), v.end()); files.insert(files.end(), v.begin(), v.end());
        } else files.push_back(p.string());
    }
    double tb = 0, te = 0, td = 0, px = 0, raw = 0; int bad = 0;
    using clk = std::chrono::steady_clock;
    for (auto& f : files) {
        RawImage img; img.load(f); img.le();
        double be = 1e30, bd = 1e30; std::vector<uint64_t> c; bool ok = true;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now(); c = llrice::compress(img); auto t1 = clk::now();
            be = std::min(be, std::chrono::duration<double>(t1 - t0).count());
            std::vector<uint64_t> cc = c;
            auto t2 = clk::now(); auto d = llrice::decompress(cc); auto t3 = clk::now();
            bd = std::min(bd, std::chrono::duration<double>(t3 - t2).count());
            if (r == 0) { RawImage ref; ref.load(f); ok = d.width == ref.width && d.height == ref.height &&
                          !memcmp(d.as<uint8_t>(), ref.as<uint8_t>(), size_t(ref.width) * ref.height * 3); }
        }
        double p = double(img.width) * img.height; px += p; raw += p * 3; tb += c.size() * 64.0; te += be; td += bd;
        if (!ok) { ++bad; printf("MISMATCH %s\n", f.c_str()); }
    }
    printf("TOTAL %zu files %.4f bpp  enc %.1f dec %.1f MB/s  %s\n", files.size(), tb / px, raw / te / 1e6, raw / td / 1e6,
           bad ? "MISMATCH" : "all ok");
}
