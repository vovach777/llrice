// Shared token extraction + binary cache for R8F5 screen training.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include "../../image.hpp"
#include "../../rlh9.hpp"

struct Tok { uint16_t sym; uint8_t vh, rc; };
struct LaneStream {
    uint8_t plane, sc;
    uint32_t fixed_bits;
    std::vector<Tok> toks;
};
struct ImgTokens {
    std::string name;
    int W, H;
    double wt;
    size_t n_bands, screen_bands;
    std::vector<LaneStream> lanes;
};

inline std::vector<std::string> list_ppms(const std::string& dir) {
    std::vector<std::string> v;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return v;
    for (auto& e : std::filesystem::directory_iterator(dir, ec))
        if (e.path().extension() == ".ppm") v.push_back(e.path().string());
    std::sort(v.begin(), v.end());
    return v;
}

// force_set < 0 => coder detector (ZTHR); otherwise forced set.
inline void extract_image(const std::string& path, int force_set, double wt,
                          const rlh9::Tables& T0, ImgTokens& out) {
    RawImage img; img.load(path); img.le();
    out.name = path; out.W = int(img.width); out.H = int(img.height);
    out.wt = wt; out.n_bands = 0; out.screen_bands = 0; out.lanes.clear();
    const uint8_t* rgb = img.as<uint8_t>();
    rlh9::Front fe(out.W);
    std::vector<uint16_t> bu(size_t(3) * rlh9::HB * out.W);
    for (int y0 = 0; y0 < out.H; y0 += rlh9::HB) {
        ++out.n_bands;
        const int nb = std::min(rlh9::HB, out.H - y0);
        const size_t PS = size_t(nb) * out.W;
        uint16_t* U0 = bu.data(); uint16_t* U1 = U0 + PS; uint16_t* US = U1 + PS;
        const size_t z = fe.band_yus(rgb, y0, nb, U0, U1, US);
        const uint32_t sc = force_set < 0
            ? uint32_t(z * 100 >= size_t(RLH9_ZTHR) * 3 * PS)
            : uint32_t(force_set);
        if (sc == 1) ++out.screen_bands;
        const int m = nb / 2;
        for (int l = 0; l < 4; ++l) {
            const int plane = l >> 1;
            const int r0 = (l & 1) ? m : 0, r1 = (l & 1) ? nb : m;
            LaneStream ls; ls.plane = uint8_t(plane); ls.sc = uint8_t(sc); ls.fixed_bits = 0;
            const uint16_t* P = plane ? US : U0;
            for (int r = r0; r < r1; ++r) {
                const uint16_t* row = P + size_t(r) * out.W;
                const uint16_t* rowu = U1 + size_t(r) * out.W;
                uint32_t rl = 0;
                for (int x = 0; x < out.W; ++x) {
                    const uint32_t v = row[x];
                    if (!v) { ++rl; continue; }
                    uint32_t rs, rm, rc, rman, cv, vm, vh, vman;
                    rlh9::enc_run_info(T0, rl, rs, rm, rc, rman);
                    rlh9::enc_val_info(T0, v, cv, vm, vh, vman);
                    ls.toks.push_back({uint16_t(rs * rlh9::NV + cv), uint8_t(vh), uint8_t(rc)});
                    ls.fixed_bits += rm + vm;
                    if (plane) {
                        const uint32_t zu = rowu[x];
                        if (v <= uint32_t(rlh9::REMS)) ls.fixed_bits += T0.rem_len[sc][v][zu];
                        else {
                            const uint32_t mm = v + 1u;
                            const uint32_t tk = 31u - uint32_t(__builtin_clz(mm));
                            const uint32_t tt = (2u << tk) - mm;
                            ls.fixed_bits += tk + uint32_t(zu >= tt);
                        }
                    }
                    rl = 0;
                }
                if (rl) ls.toks.push_back({uint16_t(rlh9::EOB), 0, 0});
            }
            out.lanes.push_back(std::move(ls));
        }
    }
}

// Binary cache: [int W][int H][int n_bands][int screen_bands][int nlanes]
//   per lane: [u8 plane][u8 sc][u32 fixed][u32 ntok][ntok x (u16 sym,u8 vh,u8 rc)]
inline bool write_cache(const std::string& path, const ImgTokens& im) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    int nl = int(im.lanes.size());
    std::fwrite(&im.W, 4, 1, f); std::fwrite(&im.H, 4, 1, f);
    int nb = int(im.n_bands), sb = int(im.screen_bands);
    std::fwrite(&nb, 4, 1, f); std::fwrite(&sb, 4, 1, f); std::fwrite(&nl, 4, 1, f);
    for (const auto& ls : im.lanes) {
        std::fwrite(&ls.plane, 1, 1, f); std::fwrite(&ls.sc, 1, 1, f);
        std::fwrite(&ls.fixed_bits, 4, 1, f);
        uint32_t n = uint32_t(ls.toks.size()); std::fwrite(&n, 4, 1, f);
        std::fwrite(ls.toks.data(), sizeof(Tok), n, f);
    }
    std::fclose(f); return true;
}

inline bool read_cache(const std::string& path, ImgTokens& im) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int nl;
    if (std::fread(&im.W,4,1,f)!=1 || std::fread(&im.H,4,1,f)!=1) { std::fclose(f); return false; }
    int nb, sb;
    std::fread(&nb,4,1,f); std::fread(&sb,4,1,f); std::fread(&nl,4,1,f);
    im.n_bands = nb; im.screen_bands = sb; im.name = path; im.wt = 0;
    im.lanes.resize(nl);
    for (int i = 0; i < nl; ++i) {
        auto& ls = im.lanes[i];
        std::fread(&ls.plane,1,1,f); std::fread(&ls.sc,1,1,f);
        std::fread(&ls.fixed_bits,4,1,f); uint32_t n; std::fread(&n,4,1,f);
        ls.toks.resize(n);
        if (n) std::fread(ls.toks.data(), sizeof(Tok), n, f);
    }
    std::fclose(f); return true;
}
