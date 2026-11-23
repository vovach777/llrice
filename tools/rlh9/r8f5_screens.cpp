// r8f5_screens.cpp - retrain ONLY the screen half (set 1) of R8F5 on new screens/scans.
// Photo half (set 0) frozen from r8f5_best48_tables.h (7e682bd).
// Band set by coder detector (force=-1, ZTHR=65). Equal total weight per group.
// Tokens read from disk cache built by build_cache.cpp -> bounded RAM (one image at a time).
#include "r8f5_common.hpp"
#include <chrono>
#include <sys/resource.h>

static const char* BASE = "/tmp/r8f5/scr";
static const char* CACHE = "/tmp/r8f5/cache";
static const int DESKTOP_STRIDE = 1;

struct CtxFsm {
    int NC;
    int init_ctx[2];
    std::vector<uint8_t> T, L;
    CtxFsm(int nc = 48) : NC(nc), T(size_t(4) * nc * 128, 0), L(size_t(4) * nc * rlh9::NSYM, 0) {
        init_ctx[0] = R8F5_INIT_CTX[0]; init_ctx[1] = R8F5_INIT_CTX[1];
    }
    inline uint8_t& trans(int sc, int pl, int ctx, int cls) { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t  trans(int sc, int pl, int ctx, int cls) const { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t* len_row(int sc, int pl, int ctx) { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }
    inline const uint8_t* len_row(int sc, int pl, int ctx) const { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }

    void load_from_header() {
        init_ctx[0] = R8F5_INIT_CTX[0]; init_ctx[1] = R8F5_INIT_CTX[1];
        for (int sc = 0; sc < 2; ++sc) for (int pl = 0; pl < 2; ++pl)
            for (int c = 0; c < NC; ++c) {
                for (int cls = 0; cls < 128; ++cls) trans(sc, pl, c, cls) = R8F5_TRANS[sc][pl][c][cls];
                int tc = sc * rlh9::SETC + pl * rlh9::NCTX + c;
                uint8_t* lr = len_row(sc, pl, c);
                for (int s = 0; s < rlh9::NSYM; ++s) {
                    uint8_t pk = R8F5_LEN[tc][s >> 1];
                    lr[s] = (s & 1) ? (pk >> 4) : (pk & 15);
                }
            }
    }

    void train_huff_screen(const std::vector<std::pair<std::string,double>>& files) {
        std::vector<double> cnt(size_t(4) * NC * rlh9::NSYM, 0.0);
        ImgTokens im;
        for (auto& fp : files) {
            if (!read_cache(fp.first, im)) { std::printf("MISSING cache %s\n", fp.first.c_str()); continue; }
            for (const auto& ls : im.lanes) {
                if (ls.sc != 1) continue;
                const int pl = ls.plane;
                int ctx = init_ctx[1];
                const size_t base = (size_t(2) + pl) * NC;
                for (const auto& t : ls.toks) {
                    cnt[(base + ctx) * rlh9::NSYM + t.sym] += fp.second;
                    if (t.sym != rlh9::EOB) ctx = trans(1, pl, ctx, (int(t.rc) << 4) | t.vh);
                }
            }
        }
        for (int pl = 0; pl < 2; ++pl)
            for (int c = 0; c < NC; ++c)
                rlh9::huff_len(&cnt[((size_t(2) + pl) * NC + c) * rlh9::NSYM],
                               rlh9::NSYM, rlh9::MAXL, len_row(1, pl, c));
    }

    void em_step_screen(const std::vector<std::pair<std::string,double>>& files, double min_wt = 25.0) {
        const size_t H = size_t(NC) * 128;
        std::vector<float> nxt(2 * H * rlh9::NSYM, 0.0f);
        std::vector<double> tot(2 * H, 0.0);
        ImgTokens im;
        for (auto& fp : files) {
            if (!read_cache(fp.first, im)) continue;
            const float wt = float(fp.second);
            for (const auto& ls : im.lanes) {
                if (ls.sc != 1) continue;
                const int pl = ls.plane;
                const size_t q = size_t(pl);
                int ctx = init_ctx[1];
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
                    ctx = trans(1, pl, ctx, cls);
                }
            }
        }
        for (int pl = 0; pl < 2; ++pl) {
            const size_t q = size_t(pl);
            for (int c = 0; c < NC; ++c) for (int cls = 0; cls < 128; ++cls) {
                const size_t key = q * H + size_t(c) * 128 + cls;
                if (tot[key] < min_wt) continue;
                const float* hist = &nxt[key * rlh9::NSYM];
                std::vector<std::pair<int,float>> nz;
                for (int s = 0; s < rlh9::NSYM; ++s) if (hist[s] > 0.0f) nz.push_back({s, hist[s]});
                if (nz.empty()) continue;
                int best_c = trans(1, pl, c, cls);
                double best_cost = 1e30;
                for (int c2 = 0; c2 < NC; ++c2) {
                    const uint8_t* L2 = len_row(1, pl, c2);
                    double cost = 0.0;
                    for (auto& p : nz) cost += double(p.second) * L2[p.first];
                    if (cost < best_cost) { best_cost = cost; best_c = c2; }
                }
                trans(1, pl, c, cls) = uint8_t(best_c);
            }
        }
        train_huff_screen(files);
    }

    double image_bits(const ImgTokens& im) const {
        size_t body = im.n_bands * 16;
        for (const auto& ls : im.lanes) {
            const int sc = ls.sc, pl = ls.plane;
            int ctx = init_ctx[sc];
            uint64_t lb = ls.fixed_bits;
            for (const auto& t : ls.toks) {
                lb += len_row(sc, pl, ctx)[t.sym];
                if (t.sym != rlh9::EOB) ctx = trans(sc, pl, ctx, (int(t.rc) << 4) | t.vh);
            }
            body += size_t((lb + 7) >> 3);
        }
        return double(2 + (body + 7) / 8) * 64.0;
    }
    // stream eval: bpp over a list of cache files; weighted if wt != 0
    double eval_stream(const std::vector<std::pair<std::string,double>>& files, bool weighted) const {
        double tb = 0, tp = 0, wb = 0, wp = 0;
        ImgTokens im;
        for (auto& fp : files) {
            if (!read_cache(fp.first, im)) continue;
            double bits = image_bits(im);
            double px = double(im.W) * im.H;
            tb += bits; tp += px;
            if (weighted) { wb += fp.second * bits; wp += fp.second * px; }
        }
        return weighted ? wb / wp : tb / tp;
    }
};

static double eval_r8f4(const std::vector<std::pair<std::string,double>>& files,
                        const rlh9::Tables& T0, bool weighted) {
    const auto& M = rlh9::fsm();
    double tb = 0, tp = 0, wb = 0, wp = 0;
    ImgTokens im;
    for (auto& fp : files) {
        if (!read_cache(fp.first, im)) continue;
        size_t bb = im.n_bands * 16;
        for (const auto& ls : im.lanes) {
            uint32_t st = rlh9::ST0, cm = rlh9::ctx_m(ls.sc), cc = rlh9::ctx_c(ls.sc);
            const uint32_t gb = ls.sc * rlh9::SETC + ls.plane * rlh9::NCTX;
            uint64_t lb = ls.fixed_bits;
            for (const auto& t : ls.toks) {
                lb += T0.len[gb + rlh9::ctx_of(st, cm, cc)][t.sym];
                if (t.sym != rlh9::EOB)
                    st = uint32_t(M.v[st & 63u][t.vh]) | (uint32_t(M.r[st >> 6][t.rc]) << 6);
            }
            bb += size_t((lb + 7) >> 3);
        }
        double bits = double(2 + (bb + 7) / 8) * 64.0;
        double px = double(im.W) * im.H;
        tb += bits; tp += px;
        if (weighted) { wb += fp.second * bits; wp += fp.second * px; }
    }
    return weighted ? wb / wp : tb / tp;
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
                for (int cls = 0; cls < 128; ++cls)
                    std::fprintf(hf, "%d%s", F.trans(sc, pl, c, cls), cls < 127 ? "," : "");
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

static std::string cache_of(const std::string& ppm) {
    // /tmp/r8f5/scr/<grp>/... -> /tmp/r8f5/cache/<grp>/... with .tok
    std::string c = std::string(CACHE) + ppm.substr(std::strlen(BASE));
    auto p = std::filesystem::path(c);
    return p.replace_extension(".tok").string();
}

int main(int argc, char** argv) {
    int ITERS = argc > 1 ? atoi(argv[1]) : 12;
    auto t_start = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    const rlh9::Tables& T0 = rlh9::tables();

    struct Grp { const char* name; std::string dir; };
    std::vector<Grp> groups = {
        {"ios", std::string(BASE)+"/ios"}, {"desktop", std::string(BASE)+"/desktop"},
        {"websight", std::string(BASE)+"/websight"}, {"docs", std::string(BASE)+"/docs"},
        {"scans", std::string(BASE)+"/scans"}, {"web", std::string(BASE)+"/web"},
    };
    std::vector<std::pair<std::string,double>> train_files, hold_files;
    std::vector<std::vector<std::pair<std::string,double>>> test_files(6);
    double group_px[6] = {0,0,0,0,0,0};
    double group_w[6] = {0,0,0,0,0,0};
    std::vector<std::string> train_ppms[6];
    // Pass 1: pixel totals per group
    for (int gi = 0; gi < 6; ++gi) {
        auto all = list_ppms(groups[gi].dir + "/all");
        for (size_t i = 0; i < all.size(); ++i) {
            std::string split;
            if (gi == 5) split = (i == 0) ? "hold" : "train";
            else if (i % 5 == 4) split = "test";
            else if (i % 10 == 3) split = "hold";
            else split = "train";
            if (split == "train") train_ppms[gi].push_back(all[i]);
        }
        if (gi == 1 && DESKTOP_STRIDE > 1) {
            std::vector<std::string> s;
            for (size_t i = 0; i < train_ppms[gi].size(); ++i)
                if (i % DESKTOP_STRIDE == 0) s.push_back(train_ppms[gi][i]);
            train_ppms[gi] = s;
        }
        ImgTokens im;
        for (auto& p : train_ppms[gi]) if (read_cache(cache_of(p), im)) group_px[gi] += double(im.W) * im.H;
    }
    // raw image weight = 1/group_pixels => each group sums to 1 total weight.
    // Normalize to mean 1 over all train images so EM min_wt=25 stays meaningful.
    size_t Ntrain = 0; for (int gi=0; gi<6; ++gi) Ntrain += train_ppms[gi].size();
    double raw_sum = 0;
    for (int gi=0; gi<6; ++gi) if (group_px[gi] > 0) raw_sum += train_ppms[gi].size() / group_px[gi];
    double norm = Ntrain / raw_sum;
    for (int gi=0; gi<6; ++gi) group_w[gi] = group_px[gi] > 0 ? norm / group_px[gi] : 0.0;
    // Pass 2: build file lists
    for (int gi = 0; gi < 6; ++gi) {
        double w = group_w[gi];
        for (auto& p : train_ppms[gi]) train_files.push_back({cache_of(p), w});
        int nt = 0, nh = 0;
        auto all = list_ppms(groups[gi].dir + "/all");
        for (size_t i = 0; i < all.size(); ++i) {
            std::string split;
            if (gi == 5) split = (i == 0) ? "hold" : "train";
            else if (i % 5 == 4) split = "test";
            else if (i % 10 == 3) split = "hold";
            else split = "train";
            std::string cp = cache_of(all[i]);
            if (split == "test") { test_files[gi].push_back({cp, 1.0}); ++nt; }
            else if (split == "hold") { hold_files.push_back({cp, w}); ++nh; }
        }
        std::printf("GROUP %-9s files=%zu train=%zu hold=%d test=%d train_px=%.0f w=%.9f\n",
                    groups[gi].name, all.size(), train_ppms[gi].size(), nh, nt, group_px[gi], w);
    }
    std::printf("TRAIN files=%zu HOLD files=%zu\n", train_files.size(), hold_files.size());
    auto fixed = [&](const char* sub) {
        std::vector<std::pair<std::string,double>> v;
        for (auto& f : list_ppms(std::string(BASE)+"/"+sub)) v.push_back({cache_of(f), 1.0});
        return v;
    };
    auto t_full12 = fixed("full12"), t_web8 = fixed("web8"), t_valid = fixed("valid"), t_kodak = fixed("kodak");
    std::printf("FIXED full12=%zu web8=%zu valid=%zu kodak=%zu\n", t_full12.size(), t_web8.size(), t_valid.size(), t_kodak.size());
    std::fflush(stdout);

    CtxFsm F0; F0.load_from_header();
    auto report = [&](const char* tag, const CtxFsm& f) {
        std::printf("%s |", tag);
        for (int gi = 0; gi < 6; ++gi) {
            if (test_files[gi].empty()) { std::printf(" %s=n/a", groups[gi].name); continue; }
            std::printf(" %s=%.4f", groups[gi].name, f.eval_stream(test_files[gi], false));
        }
        std::printf(" full12=%.4f web8=%.4f valid41=%.4f kodak24=%.4f hold_w=%.4f\n",
                    f.eval_stream(t_full12,false), f.eval_stream(t_web8,false),
                    f.eval_stream(t_valid,false), f.eval_stream(t_kodak,false),
                    f.eval_stream(hold_files,true));
        std::fflush(stdout);
    };
    std::printf("[%.1fs] R8F4(1c440de) |", secs());
    for (int gi = 0; gi < 6; ++gi) {
        if (test_files[gi].empty()) { std::printf(" %s=n/a", groups[gi].name); continue; }
        std::printf(" %s=%.4f", groups[gi].name, eval_r8f4(test_files[gi], T0, false));
    }
    std::printf(" full12=%.4f web8=%.4f valid41=%.4f kodak24=%.4f hold_w=%.4f\n",
                eval_r8f4(t_full12,T0,false), eval_r8f4(t_web8,T0,false), eval_r8f4(t_valid,T0,false),
                eval_r8f4(t_kodak,T0,false), eval_r8f4(hold_files,T0,true));
    std::fflush(stdout);
    std::printf("[%.1fs] R8F5(7e682bd) |", secs());
    report("     ", F0);

    CtxFsm F; F.load_from_header();
    double best_hold = F.eval_stream(hold_files, true);
    CtxFsm bestF = F; int best_iter = 0;
    for (int it = 1; it <= ITERS; ++it) {
        std::printf("[%.1fs] EM iteration %d ... ", secs(), it); std::fflush(stdout);
        F.em_step_screen(train_files, 25.0);
        char tag[32]; std::snprintf(tag, sizeof tag, "EM%-2d", it);
        report(tag, F);
        double h = F.eval_stream(hold_files, true);
        if (h < best_hold) { best_hold = h; bestF = F; best_iter = it; }
        std::printf("  [%6.1fs] hold=%.4f best=%d(%.4f)\n", secs(), h, best_iter, best_hold);
        std::fflush(stdout);
    }
    std::printf("\nBEST iteration=%d (weighted screen holdout) hold=%.4f\n", best_iter, best_hold);
    report("BEST", bestF);

    // screen-band fraction per group (train)
    ImgTokens im;
    for (int gi = 0; gi < 6; ++gi) {
        size_t sb = 0, nb = 0;
        for (auto& p : train_ppms[gi]) if (read_cache(cache_of(p), im)) { sb += im.screen_bands; nb += im.n_bands; }
        std::printf("SCREEN_FRAC %-9s train bands %zu/%zu = %.2f%%\n", groups[gi].name, sb, nb, nb ? 100.0*sb/nb : 0.0);
    }
    // band fraction on test sets too (for scans/docs insight)
    for (int gi = 0; gi < 6; ++gi) {
        size_t sb = 0, nb = 0;
        for (auto& fp : test_files[gi]) if (read_cache(fp.first, im)) { sb += im.screen_bands; nb += im.n_bands; }
        std::printf("SCREEN_FRAC %-9s TEST  bands %zu/%zu = %.2f%%\n", groups[gi].name, sb, nb, nb ? 100.0*sb/nb : 0.0);
    }
    // photo-control band fractions (bands the detector sends to screen on valid41/kodak24)
    {
        auto frac = [&](const std::vector<std::pair<std::string,double>>& fs, const char* nm) {
            size_t sb = 0, nb = 0;
            for (auto& fp : fs) if (read_cache(fp.first, im)) { sb += im.screen_bands; nb += im.n_bands; }
            std::printf("SCREEN_FRAC %-9s       bands %zu/%zu = %.2f%% -> encoded by new screen tables\n", nm, sb, nb, nb ? 100.0*sb/nb : 0.0);
        };
        frac(t_valid, "valid41"); frac(t_kodak, "kodak24"); frac(t_full12, "full12"); frac(t_web8, "web8");
    }
    std::fflush(stdout);

    FILE* hf = std::fopen("/tmp/r8f5/r8f5_screens_best_tables.h", "w");
    export_header(hf, bestF); std::fclose(hf);
    std::printf("Exported /tmp/r8f5/r8f5_screens_best_tables.h\n");
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    std::printf("TIME %.1f s  MAXRSS %.1f MB\n", secs(), ru.ru_maxrss/1024.0);
    return 0;
}
