// r8f5_stream.cpp - streaming R8F5 trainer for the full corpus.
// Only counters in RAM; images are re-read from disk on each pass.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <sys/resource.h>
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
    size_t n_bands;
    std::vector<LaneStream> lanes;
};

static std::vector<std::string> list_ppms(const std::string& dir) {
    std::vector<std::string> v;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".ppm") v.push_back(e.path().string());
    std::sort(v.begin(), v.end());
    return v;
}

static void extract_image(const std::string& path, int force_set, double wt,
                          const rlh9::Tables& T0, ImgTokens& out) {
    RawImage img; img.load(path); img.le();
    out.name = path; out.W = int(img.width); out.H = int(img.height);
    out.wt = wt; out.n_bands = 0; out.lanes.clear();
    const uint8_t* rgb = img.as<uint8_t>();
    rlh9::Front fe(out.W);
    std::vector<uint16_t> bu(size_t(3) * rlh9::HB * out.W);
    for (int y0 = 0; y0 < out.H; y0 += rlh9::HB) {
        ++out.n_bands;
        const int nb = std::min(rlh9::HB, out.H - y0);
        const size_t PS = size_t(nb) * out.W;
        uint16_t* U0 = bu.data(); uint16_t* U1 = U0 + PS; uint16_t* US = U1 + PS;
        const size_t z = fe.band_yus(rgb, y0, nb, U0, U1, US);
        const uint32_t sc = force_set < 0 ? uint32_t(z * 100 >= size_t(RLH9_ZTHR) * 3 * PS) : uint32_t(force_set);
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

struct CtxFsm {
    int NC;
    int init_ctx[2];
    std::vector<uint8_t> T;
    std::vector<uint8_t> L;
    CtxFsm(int nc) : NC(nc), T(size_t(4) * nc * 128, 0), L(size_t(4) * nc * rlh9::NSYM, 0) {
        init_ctx[0] = 0; init_ctx[1] = 0;
    }
    inline uint8_t& trans(int sc, int pl, int ctx, int cls) { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t  trans(int sc, int pl, int ctx, int cls) const { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t* len_row(int sc, int pl, int ctx) { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }
    inline const uint8_t* len_row(int sc, int pl, int ctx) const { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }

    void train_huff_stream(const std::vector<std::pair<std::string,double>>& files,
                           const std::vector<int>& forces, const rlh9::Tables& T0) {
        std::vector<double> cnt(size_t(4) * NC * rlh9::NSYM, 0.0);
        for (size_t fi = 0; fi < files.size(); ++fi) {
            ImgTokens im; extract_image(files[fi].first, forces[fi], files[fi].second, T0, im);
            for (const auto& ls : im.lanes) {
                const int sc = ls.sc, pl = ls.plane;
                int ctx = init_ctx[sc];
                const size_t base = (size_t(sc) * 2 + pl) * NC;
                for (const auto& t : ls.toks) {
                    cnt[(base + ctx) * rlh9::NSYM + t.sym] += im.wt;
                    if (t.sym != rlh9::EOB) ctx = trans(sc, pl, ctx, (int(t.rc) << 4) | t.vh);
                }
            }
        }
        for (int sc = 0; sc < 2; ++sc)
            for (int pl = 0; pl < 2; ++pl)
                for (int c = 0; c < NC; ++c)
                    rlh9::huff_len(&cnt[((size_t(sc) * 2 + pl) * NC + c) * rlh9::NSYM], rlh9::NSYM, rlh9::MAXL, len_row(sc, pl, c));
    }

    void em_step_stream(const std::vector<std::pair<std::string,double>>& files,
                        const std::vector<int>& forces, const rlh9::Tables& T0,
                        double min_wt = 25.0) {
        const size_t H = size_t(NC) * 128;
        std::vector<float> nxt(4 * H * rlh9::NSYM, 0.0f);
        std::vector<double> tot(4 * H, 0.0);
        for (size_t fi = 0; fi < files.size(); ++fi) {
            ImgTokens im; extract_image(files[fi].first, forces[fi], files[fi].second, T0, im);
            const float wt = float(im.wt);
            for (const auto& ls : im.lanes) {
                const int sc = ls.sc, pl = ls.plane;
                const size_t q = size_t(sc) * 2 + pl;
                int ctx = init_ctx[sc];
                const size_t nt = ls.toks.size();
                for (size_t i = 0; i < nt; ++i) {
                    const auto& t = ls.toks[i];
                    if (t.sym == rlh9::EOB) continue;
                    const int cls = (int(t.rc) << 4) | t.vh;
                    if (i + 1 < nt) {
                        const size_t key = q * H + size_t(ctx) * 128 + cls;
                        nxt[key * rlh9::NSYM + ls.toks[i + 1].sym] += wt;
                        tot[key] += wt;
                    }
                    ctx = trans(sc, pl, ctx, cls);
                }
            }
        }
        for (int sc = 0; sc < 2; ++sc) for (int pl = 0; pl < 2; ++pl) {
            const size_t q = size_t(sc) * 2 + pl;
            for (int c = 0; c < NC; ++c) for (int cls = 0; cls < 128; ++cls) {
                const size_t key = q * H + size_t(c) * 128 + cls;
                if (tot[key] < min_wt) continue;
                const float* hist = &nxt[key * rlh9::NSYM];
                std::vector<std::pair<int,float>> nz;
                for (int s = 0; s < rlh9::NSYM; ++s) if (hist[s] > 0.0f) nz.push_back({s, hist[s]});
                int best_c = trans(sc, pl, c, cls);
                double best_cost = 1e30;
                for (int c2 = 0; c2 < NC; ++c2) {
                    const uint8_t* L2 = len_row(sc, pl, c2);
                    double cost = 0.0;
                    for (auto& p : nz) cost += double(p.second) * L2[p.first];
                    if (cost < best_cost) { best_cost = cost; best_c = c2; }
                }
                trans(sc, pl, c, cls) = uint8_t(best_c);
            }
        }
        train_huff_stream(files, forces, T0);
    }

    double eval_tokens(const std::vector<ImgTokens>& imgs) const {
        double total_bits = 0, total_px = 0;
        for (const auto& im : imgs) {
            size_t body_bytes = im.n_bands * 16;
            for (const auto& ls : im.lanes) {
                const int sc = ls.sc, pl = ls.plane;
                int ctx = init_ctx[sc];
                uint64_t lbits = ls.fixed_bits;
                for (const auto& t : ls.toks) {
                    lbits += len_row(sc, pl, ctx)[t.sym];
                    if (t.sym != rlh9::EOB) ctx = trans(sc, pl, ctx, (int(t.rc) << 4) | t.vh);
                }
                body_bytes += size_t((lbits + 7) >> 3);
            }
            const size_t words = 2 + (body_bytes + 7) / 8;
            total_bits += double(words) * 64.0;
            total_px += double(im.W) * im.H;
        }
        return total_bits / total_px;
    }

    double eval_stream(const std::vector<std::pair<std::string,double>>& files,
                       const std::vector<int>& forces, const rlh9::Tables& T0) const {
        double tb = 0, tp = 0;
        for (size_t fi = 0; fi < files.size(); ++fi) {
            ImgTokens im; extract_image(files[fi].first, forces[fi], files[fi].second, T0, im);
            size_t bb = im.n_bands * 16;
            for (const auto& ls : im.lanes) {
                const int sc = ls.sc, pl = ls.plane;
                int ctx = init_ctx[sc];
                uint64_t lb = ls.fixed_bits;
                for (const auto& t : ls.toks) {
                    lb += len_row(sc, pl, ctx)[t.sym];
                    if (t.sym != rlh9::EOB) ctx = trans(sc, pl, ctx, (int(t.rc) << 4) | t.vh);
                }
                bb += size_t((lb + 7) >> 3);
            }
            tb += double(2 + (bb + 7) / 8) * 64.0;
            tp += double(im.W) * im.H;
        }
        return tb / tp;
    }
};

static int r8f4_proj(int sc, int sv, int sr, int NC) {
    if (NC == 48) return sc ? ((sv >> 2) * 4 + (sr >> 2)) : ((sv >> 1) * 2 + (sr >> 3));
    if (NC == 64) {
        if (sc) return ((sv * 16) / 48) * 4 + (sr >> 2);
        return ((sv * 32) / 48) * 2 + (sr >> 3);
    }
    if (NC == 96) return sc ? ((sv >> 1) * 4 + (sr >> 2)) : (sv * 2 + (sr >> 3));
    return 0;
}

// Method 1: majority quantization of R8F4 transitions (streaming)
static CtxFsm build_method1_quant(int NC,
        const std::vector<std::pair<std::string,double>>& files,
        const std::vector<int>& forces, const rlh9::Tables& T0) {
    CtxFsm F(NC);
    const auto& M = rlh9::fsm();
    F.init_ctx[0] = r8f4_proj(0, 12, 0, NC);
    F.init_ctx[1] = r8f4_proj(1, 12, 0, NC);
    std::vector<double> votes(size_t(4) * NC * 128 * NC, 0.0);
    for (size_t fi = 0; fi < files.size(); ++fi) {
        ImgTokens im; extract_image(files[fi].first, forces[fi], files[fi].second, T0, im);
        const double wt = im.wt;
        for (const auto& ls : im.lanes) {
            const int sc = ls.sc, pl = ls.plane;
            uint32_t st = rlh9::ST0;
            for (const auto& t : ls.toks) {
                if (t.sym == rlh9::EOB) continue;
                const int sv = int(st & 63u), sr = int(st >> 6);
                const int c = r8f4_proj(sc, sv, sr, NC);
                const int cls = (int(t.rc) << 4) | t.vh;
                const uint32_t nst = uint32_t(M.v[sv][t.vh]) | (uint32_t(M.r[sr][t.rc]) << 6);
                const int nc = r8f4_proj(sc, int(nst & 63u), int(nst >> 6), NC);
                votes[(((size_t(sc) * 2 + pl) * NC + c) * 128 + cls) * NC + nc] += wt;
                st = nst;
            }
        }
    }
    for (int sc = 0; sc < 2; ++sc) for (int pl = 0; pl < 2; ++pl) for (int c = 0; c < NC; ++c) {
        int sr_bins = sc ? 4 : 2;
        int sv_bins = NC / sr_bins;
        int vb = c / sr_bins, rb = c % sr_bins;
        int rep_sv = std::min(47, (vb * 48 + 24) / sv_bins);
        int rep_sr = std::min(15, (rb * 16 + 8) / sr_bins);
        for (int cls = 0; cls < 128; ++cls) {
            const int vh = cls & 15, rc = cls >> 4;
            const double* vrow = &votes[(((size_t(sc) * 2 + pl) * NC + c) * 128 + cls) * NC];
            int best_nc = -1; double best_v = 0.0;
            for (int nc = 0; nc < NC; ++nc) if (vrow[nc] > best_v) { best_v = vrow[nc]; best_nc = nc; }
            if (best_nc < 0) {
                int nsv = M.v[rep_sv][vh], nsr = M.r[rep_sr][rc];
                best_nc = r8f4_proj(sc, nsv, nsr, NC);
            }
            F.trans(sc, pl, c, cls) = uint8_t(best_nc);
        }
    }
    F.train_huff_stream(files, forces, T0);
    return F;
}

static double eval_r8f4_stream(const std::vector<std::pair<std::string,double>>& files,
                               const std::vector<int>& forces, const rlh9::Tables& T0) {
    const auto& M = rlh9::fsm();
    double tb = 0, tp = 0;
    for (size_t fi = 0; fi < files.size(); ++fi) {
        ImgTokens im; extract_image(files[fi].first, forces[fi], files[fi].second, T0, im);
        size_t bb = im.n_bands * 16;
        for (const auto& ls : im.lanes) {
            uint32_t st = rlh9::ST0, cm = rlh9::ctx_m(ls.sc), cc = rlh9::ctx_c(ls.sc);
            const uint32_t gb = ls.sc * rlh9::SETC + ls.plane * rlh9::NCTX;
            uint64_t lb = ls.fixed_bits;
            for (const auto& t : ls.toks) {
                lb += T0.len[gb + rlh9::ctx_of(st, cm, cc)][t.sym];
                if (t.sym != rlh9::EOB) st = uint32_t(M.v[st & 63u][t.vh]) | (uint32_t(M.r[st >> 6][t.rc]) << 6);
            }
            bb += size_t((lb + 7) >> 3);
        }
        tb += double(2 + (bb + 7) / 8) * 64.0;
        tp += double(im.W) * im.H;
    }
    return tb / tp;
}

static void export_header(FILE* hf, const CtxFsm& F) {
    std::fprintf(hf, "#pragma once\n#include <cstdint>\n");
    std::fprintf(hf, "static const uint8_t R8F5_INIT_CTX[2] = {%d, %d};\n", F.init_ctx[0], F.init_ctx[1]);
    std::fprintf(hf, "static const uint8_t R8F5_TRANS[2][2][%d][128] = {\n", F.NC);
    for (int sc = 0; sc < 2; ++sc) {
        std::fprintf(hf, "{\n");
        for (int pl = 0; pl < 2; ++pl) {
            std::fprintf(hf, "{\n");
            for (int c = 0; c < F.NC; ++c) {
                std::fprintf(hf, "{");
                for (int cls = 0; cls < 128; ++cls) std::fprintf(hf, "%d%s", F.trans(sc, pl, c, cls), cls < 127 ? "," : "");
                std::fprintf(hf, "}%s\n", c + 1 < F.NC ? "," : "");
            }
            std::fprintf(hf, "}%s\n", pl < 1 ? "," : "");
        }
        std::fprintf(hf, "}%s\n", sc < 1 ? "," : "");
    }
    std::fprintf(hf, "};\n");
    constexpr int PB = (rlh9::NSYM + 1) / 2;
    std::fprintf(hf, "alignas(64) static const uint8_t R8F5_LEN[%d][%d] = {\n", rlh9::TOTC, PB);
    for (int tc = 0; tc < rlh9::TOTC; ++tc) {
        int sc = tc / rlh9::SETC, pl = (tc % rlh9::SETC) / 48, c = tc % 48;
        const uint8_t* Lrow = F.len_row(sc, pl, c);
        std::fprintf(hf, "{");
        for (int b = 0; b < PB; ++b) {
            int lo = Lrow[2 * b], hi = (2 * b + 1 < rlh9::NSYM) ? Lrow[2 * b + 1] : 0;
            std::fprintf(hf, "%d%s", lo | (hi << 4), b + 1 < PB ? "," : "");
        }
        std::fprintf(hf, "}%s\n", tc + 1 < rlh9::TOTC ? "," : "");
    }
    std::fprintf(hf, "};\n");
}

int main(int argc, char** argv) {
    const char* P = "/tmp/r8f5/prep";
    int ITERS = argc > 1 ? atoi(argv[1]) : 8;
    auto t_start = std::chrono::steady_clock::now();
    const rlh9::Tables& T0 = rlh9::tables();   // committed tables from 1c440de

    // group list: every 10th file (alphabetical index 9,19,...) is held out
    struct Grp { const char* name; std::string dir; int force; double wt; };
    std::vector<Grp> groups = {
        {"photo", std::string(P)+"/photo", 0, 1.0},
        {"gb82",  std::string(P)+"/gb82",  1, 4.0},
        {"web",   std::string(P)+"/web",   1, 4.0},
        {"game",  std::string(P)+"/game",  1, 1.0},
    };
    std::vector<std::pair<std::string,double>> train_files, hold_files;
    std::vector<int> train_forces, hold_forces;
    // exact stems from train_list.txt (585 photo, 4 gb82, 8 web, 617 game)
    std::vector<std::string> want[4];
    {
        FILE* f = std::fopen("/tmp/r8f5/llrice/tools/rlh9/train_list.txt", "r");
        char line[512];
        while (f && std::fgets(line, sizeof line, f)) {
            if (line[0] == '#') continue;
            char g[32], s[256];
            if (std::sscanf(line, "%31s\t%255s", g, s) != 2) continue;
            int gi = !std::strcmp(g,"photo") ? 0 : !std::strcmp(g,"gb82") ? 1 : !std::strcmp(g,"web") ? 2 : !std::strcmp(g,"game") ? 3 : -1;
            if (gi >= 0) want[gi].push_back(s);
        }
        if (f) std::fclose(f);
    }
    for (int gi = 0; gi < 4; ++gi) std::sort(want[gi].begin(), want[gi].end());
    for (int gi = 0; gi < 4; ++gi) {
        auto& g = groups[gi];
        size_t nh = want[gi].size() / 10; if (nh == 0 && !want[gi].empty()) nh = 1;
        size_t ntrain = 0, nhold = 0;
        for (size_t i = 0; i < want[gi].size(); ++i) {
            std::string path = std::string(P) + "/" + g.name + "/" + want[gi][i] + ".ppm";
            if (!std::filesystem::exists(path)) { std::printf("MISSING %s\n", path.c_str()); continue; }
            bool held;
            if (want[gi].size() < 10) held = (i == 0);
            else held = ((i + 1) % 10 == 0 && nhold < nh);
            if (held) { hold_files.push_back({path, g.wt}); hold_forces.push_back(g.force); ++nhold; }
            else      { train_files.push_back({path, g.wt}); train_forces.push_back(g.force); ++ntrain; }
        }
        std::printf("GROUP %s: listed=%zu train=%zu holdout=%zu weight=%.1f force=%d\n", g.name, want[gi].size(), ntrain, nhold, g.wt, g.force);
    }
    std::printf("TRAIN files=%zu  HOLDOUT files=%zu\n", train_files.size(), hold_files.size());

    // test sets: valid41 (photos), kodak24 (photos), full12 (screens)
    std::vector<ImgTokens> test_photos, test_screens;
    auto load_test = [&](const std::string& dir, int force, std::vector<ImgTokens>& dst) {
        for (auto& f : list_ppms(dir)) { ImgTokens im; extract_image(f, force, 1.0, T0, im); dst.push_back(std::move(im)); }
    };
    load_test(std::string(P)+"/valid", 0, test_photos);
    load_test(std::string(P)+"/kodak", 0, test_photos);
    load_test(std::string(P)+"/full12", 1, test_screens);
    std::printf("TEST photos=%zu screens=%zu\n", test_photos.size(), test_screens.size());

    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    std::printf("[%.1fs] R8F4 baseline (committed 1c440de tables)\n", secs());
    auto fs_pairs = [](const std::string& dir) {
        auto v = list_ppms(dir); std::vector<std::pair<std::string,double>> r;
        for (auto& f : v) r.push_back({f,1.0}); return r;
    };
    auto vf = fs_pairs(std::string(P)+"/valid"), kf = fs_pairs(std::string(P)+"/kodak"), sf = fs_pairs(std::string(P)+"/full12");
    std::vector<int> vg(vf.size(),0), kg(kf.size(),0), sg(sf.size(),1);
    double b4_valid = eval_r8f4_stream(vf, vg, T0);
    double b4_kodak = eval_r8f4_stream(kf, kg, T0);
    double b4_full  = eval_r8f4_stream(sf, sg, T0);
    double b4_hold  = eval_r8f4_stream(hold_files, hold_forces, T0);
    std::printf("R8F4 | valid41=%.4f kodak24=%.4f full12=%.4f holdout=%.4f\n", b4_valid, b4_kodak, b4_full, b4_hold);
    std::fflush(stdout);

    const int NC = 48;
    std::printf("[%.1fs] Method 1: majority quantization of R8F4 transitions\n", secs());
    CtxFsm F = build_method1_quant(NC, train_files, train_forces, T0);
    auto show = [&](const char* label, const CtxFsm& f) {
        double v = f.eval_tokens(test_photos), k = 0;
        // split photos by evaluating separately (valid / kodak) using in-memory tokens
        // recompute with subsets for clarity
        std::vector<ImgTokens> ph(std::move(test_photos));
        (void)ph; (void)k; (void)v;
    };
    (void)show;
    // keep separate test token sets for per-set numbers
    std::vector<ImgTokens> t_valid, t_kodak;
    load_test(std::string(P)+"/valid", 0, t_valid);
    load_test(std::string(P)+"/kodak", 0, t_kodak);

    auto evaluate = [&](const CtxFsm& f, double& v, double& k, double& s, double& h) {
        v = f.eval_tokens(t_valid); k = f.eval_tokens(t_kodak); s = f.eval_tokens(test_screens);
        h = f.eval_stream(hold_files, hold_forces, T0);
    };
    double v0,k0,s0,h0; evaluate(F,v0,k0,s0,h0);
    std::printf("M1   | valid41=%.4f kodak24=%.4f full12=%.4f holdout=%.4f\n", v0,k0,s0,h0);
    std::fflush(stdout);

    double best_hold = h0; CtxFsm bestF = F; int best_iter = 0;
    for (int it = 1; it <= ITERS; ++it) {
        std::printf("[%.1fs] EM iteration %d: statistics pass\n", secs(), it); std::fflush(stdout);
        F.em_step_stream(train_files, train_forces, T0, 25.0);
        double v,k,s,h; evaluate(F,v,k,s,h);
        std::printf("EM%-2d | valid41=%.4f kodak24=%.4f full12=%.4f holdout=%.4f\n", it, v,k,s,h);
        std::printf("  delta vs R8F4: valid=%+.3f%% kodak=%+.3f%% full12=%+.3f%% hold=%+.3f%%\n",
                    (v-b4_valid)/b4_valid*100, (k-b4_kodak)/b4_kodak*100,
                    (s-b4_full)/b4_full*100, (h-b4_hold)/b4_hold*100);
        std::fflush(stdout);
        if (h < best_hold) { best_hold = h; bestF = F; best_iter = it; }
    }
    std::printf("\nBEST iteration=%d (selected by HOLDOUT bpp, not test) holdout=%.4f\n", best_iter, best_hold);
    std::printf("BEST | valid41=%.4f kodak24=%.4f full12=%.4f holdout=%.4f\n",
                bestF.eval_tokens(t_valid), bestF.eval_tokens(t_kodak),
                bestF.eval_tokens(test_screens), best_hold);
    FILE* hf = std::fopen("/tmp/r8f5/r8f5_best48_tables.h","w"); export_header(hf,bestF); std::fclose(hf);
    std::printf("Exported /tmp/r8f5/r8f5_best48_tables.h\n");
    struct rusage ru; getrusage(RUSAGE_SELF,&ru);
    std::printf("TIME %.1f s  MAXRSS %.1f MB\n", secs(), ru.ru_maxrss/1024.0);
    return 0;
}
