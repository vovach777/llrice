// r8f5_exp.cpp - Part B (R8F5: state == context) bpp study:
// Compares Method 1 (R8F4 majority quantization), Method 2 (exact cross-entropy EM/k-means),
// Method 3 (48-state direct hysteresis), and Q4 (64 / 96 states) vs R8F4 baseline.
// Also exports the best R8F5 transition table T_best.h for Part B speed benchmarking!
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
                          const rlh9::Tables& T0, ImgTokens& out,
                          std::vector<double>* rem_cnt = nullptr) {
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
                        if (v <= uint32_t(rlh9::REMS)) {
                            if (rem_cnt) (*rem_cnt)[(size_t(sc) * (rlh9::REMS + 1) + v) * (rlh9::REMS + 1) + zu] += wt;
                            else ls.fixed_bits += T0.rem_len[sc][v][zu];
                        } else {
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

// Generic N-state context-only FSM: T[sc][plane][ctx][cls] -> next_ctx in 0..NC-1, where cls = (rc << 4) | vh (0..127)
struct CtxFsm {
    int NC; // 48, 64, or 96
    int init_ctx[2]; // initial context at start of lane for sc=0,1
    std::vector<uint8_t> T; // [2][2][NC][128]
    std::vector<uint8_t> L; // [2][2][NC][NSYM]

    CtxFsm(int nc) : NC(nc), T(size_t(4) * nc * 128, 0), L(size_t(4) * nc * rlh9::NSYM, 0) {
        init_ctx[0] = 0; init_ctx[1] = 0;
    }
    inline uint8_t& trans(int sc, int pl, int ctx, int cls) { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t  trans(int sc, int pl, int ctx, int cls) const { return T[((size_t(sc) * 2 + pl) * NC + ctx) * 128 + cls]; }
    inline uint8_t* len_row(int sc, int pl, int ctx) { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }
    inline const uint8_t* len_row(int sc, int pl, int ctx) const { return &L[((size_t(sc) * 2 + pl) * NC + ctx) * rlh9::NSYM]; }

    void train_huff(const std::vector<ImgTokens>& train_imgs) {
        std::vector<double> cnt(size_t(4) * NC * rlh9::NSYM, 0.0);
        for (const auto& im : train_imgs) {
            const double wt = im.wt;
            for (const auto& ls : im.lanes) {
                const int sc = ls.sc, pl = ls.plane;
                int ctx = init_ctx[sc];
                const size_t base = (size_t(sc) * 2 + pl) * NC;
                for (const auto& t : ls.toks) {
                    cnt[(base + ctx) * rlh9::NSYM + t.sym] += wt;
                    if (t.sym != rlh9::EOB) ctx = trans(sc, pl, ctx, (int(t.rc) << 4) | t.vh);
                }
            }
        }
        for (int sc = 0; sc < 2; ++sc)
            for (int pl = 0; pl < 2; ++pl)
                for (int c = 0; c < NC; ++c)
                    rlh9::huff_len(&cnt[((size_t(sc) * 2 + pl) * NC + c) * rlh9::NSYM], rlh9::NSYM, rlh9::MAXL, len_row(sc, pl, c));
    }

    double eval_bpp(const std::vector<ImgTokens>& imgs) const {
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

    // Method 2: One EM / coordinate-descent step on T (exact cross-entropy minimization on next token!)
    void em_step(const std::vector<ImgTokens>& train_imgs, double min_wt = 25.0) {
        // Group transitions by (sc, pl, ctx, cls) -> sparse or flat histogram over NSYM?
        // 4 * NC * 128 * 837 floats would be ~160 MB (fits easily in RAM, or do per (sc, pl) = 40 MB!)
        for (int sc = 0; sc < 2; ++sc) {
            for (int pl = 0; pl < 2; ++pl) {
                std::vector<float> nxt(size_t(NC) * 128 * rlh9::NSYM, 0.0f);
                std::vector<double> tot(size_t(NC) * 128, 0.0);
                for (const auto& im : train_imgs) {
                    const float wt = float(im.wt);
                    for (const auto& ls : im.lanes) {
                        if (ls.sc != sc || ls.plane != pl) continue;
                        int ctx = init_ctx[sc];
                        const size_t nt = ls.toks.size();
                        for (size_t i = 0; i < nt; ++i) {
                            const auto& t = ls.toks[i];
                            if (t.sym == rlh9::EOB) continue;
                            const int cls = (int(t.rc) << 4) | t.vh;
                            if (i + 1 < nt) {
                                const size_t key = size_t(ctx) * 128 + cls;
                                nxt[key * rlh9::NSYM + ls.toks[i + 1].sym] += wt;
                                tot[key] += wt;
                            }
                            ctx = trans(sc, pl, ctx, cls);
                        }
                    }
                }
                // Pick best next_ctx in 0..NC-1 minimizing expected bit length of next symbol
                for (int c = 0; c < NC; ++c) {
                    for (int cls = 0; cls < 128; ++cls) {
                        const size_t key = size_t(c) * 128 + cls;
                        if (tot[key] < min_wt) continue;
                        const float* hist = &nxt[key * rlh9::NSYM];
                        // Collect non-zero symbols for fast dot product
                        std::vector<std::pair<int, float>> nz;
                        for (int s = 0; s < rlh9::NSYM; ++s) if (hist[s] > 0.0f) nz.push_back({s, hist[s]});
                        int best_c = trans(sc, pl, c, cls);
                        double best_cost = 1e30;
                        for (int c2 = 0; c2 < NC; ++c2) {
                            const uint8_t* L2 = len_row(sc, pl, c2);
                            double cost = 0.0;
                            for (auto [s, w] : nz) cost += double(w) * L2[s];
                            if (cost < best_cost) { best_cost = cost; best_c = c2; }
                        }
                        trans(sc, pl, c, cls) = uint8_t(best_c);
                    }
                }
            }
        }
        train_huff(train_imgs);
    }
};

// Map R8F4 (sv in 0..47, sr in 0..15) -> ctx in 0..NC-1
static int r8f4_proj(int sc, int sv, int sr, int NC) {
    if (NC == 48) return sc ? ((sv >> 2) * 4 + (sr >> 2)) : ((sv >> 1) * 2 + (sr >> 3));
    if (NC == 64) {
        if (sc) return ((sv * 16) / 48) * 4 + (sr >> 2); // 16x4
        return ((sv * 32) / 48) * 2 + (sr >> 3);         // 32x2
    }
    if (NC == 96) return sc ? ((sv >> 1) * 4 + (sr >> 2)) : (sv * 2 + (sr >> 3)); // 24x4 screens, 48x2 photos
    return 0;
}

// Method 1: Majority quantization of R8F4 transitions on training corpus
static CtxFsm build_method1_quant(int NC, const std::vector<ImgTokens>& train_imgs) {
    CtxFsm F(NC);
    const auto& M = rlh9::fsm();
    F.init_ctx[0] = r8f4_proj(0, 12, 0, NC);
    F.init_ctx[1] = r8f4_proj(1, 12, 0, NC);
    // Count (sc, pl, ctx, cls) -> next_ctx under R8F4
    std::vector<double> votes(size_t(4) * NC * 128 * NC, 0.0);
    for (const auto& im : train_imgs) {
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
    // Fill majority vote (fallback: analytic mid-point of bin)
    for (int sc = 0; sc < 2; ++sc) {
        for (int pl = 0; pl < 2; ++pl) {
            for (int c = 0; c < NC; ++c) {
                // Representative (sv, sr) for fallback
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
        }
    }
    F.train_huff(train_imgs);
    return F;
}

// Method 3: Direct 48-state zero-hidden-memory hysteresis automaton (24x2 photo, 12x4 screen)
static CtxFsm build_method3_hyst(const std::vector<ImgTokens>& train_imgs) {
    CtxFsm F(48);
    const auto& M = rlh9::fsm();
    F.init_ctx[0] = (12 >> 1) * 2 + 0;
    F.init_ctx[1] = (12 >> 2) * 4 + 0;
    for (int sc = 0; sc < 2; ++sc) {
        const int sr_bins = sc ? 4 : 2;
        const int sv_bins = 48 / sr_bins;
        for (int c = 0; c < 48; ++c) {
            const int vb = c / sr_bins, rb = c % sr_bins;
            const int rep_sv = sc ? (vb * 4 + 2) : (vb * 2 + 1);
            for (int cls = 0; cls < 128; ++cls) {
                const int vh = cls & 15, rc = cls >> 4;
                // sv step via R8F4 law at bin center, guaranteed to move down by >=1 bin if needed
                int nsv = M.v[rep_sv][vh];
                int nvb = sc ? (nsv >> 2) : (nsv >> 1);
                if (nvb == vb && nsv < rep_sv && vb > 0) nvb = vb - 1;
                if (nvb == vb && nsv > rep_sv && vb + 1 < sv_bins) nvb = vb + 1;
                // sr step with hysteresis
                int nrb = rb;
                if (sc == 0) {
                    // 2 bins: 0 = dense (rc=0), 1 = runs (rc>=1)
                    if (rc >= 2) nrb = 1;
                    else if (rc == 0) nrb = 0;
                } else {
                    // 4 bins: asymmetric fast-up / slow-down
                    int tgt = std::min(3, (rc * 3 + 3) / 7);
                    if (tgt > rb) nrb = std::min(3, rb + 1 + (tgt - rb >= 2 ? 1 : 0));
                    else if (tgt < rb) nrb = std::max(0, rb - 1);
                }
                const uint8_t nc = uint8_t(nvb * sr_bins + nrb);
                F.trans(sc, 0, c, cls) = nc;
                F.trans(sc, 1, c, cls) = nc;
            }
        }
    }
    F.train_huff(train_imgs);
    return F;
}

int main() {
    rlh9::Tables T0; T0.enc_luts();
    std::vector<double> rem_cnt(size_t(rlh9::NSET) * (rlh9::REMS + 1) * (rlh9::REMS + 1), 0.0);
    std::vector<ImgTokens> train_imgs, test_photos, test_screens;
    auto add_dir = [&](const std::string& dir, int force, double wt, std::vector<ImgTokens>& dst, std::vector<double>* rc) {
        for (auto& f : list_ppms(dir)) { ImgTokens im; extract_image(f, force, wt, T0, im, rc); dst.push_back(std::move(im)); }
    };
    add_dir("../rlh_v2/data/clic_train20", 0, 1.0, train_imgs, &rem_cnt);
    add_dir("../rlh_v2/data/train_gb82-sc_p6", 1, 4.0, train_imgs, &rem_cnt);
    add_dir("../rlh_v2/data/train_qoi-screenshot-web_p6", 1, 4.0, train_imgs, &rem_cnt);
    add_dir("../rlh_v2/data/train_screenshot-game_p6", 1, 1.0, train_imgs, &rem_cnt);

    uint8_t remlen[rlh9::NSET][rlh9::REMS + 1][rlh9::REMS + 1];
    std::memset(remlen, 0, sizeof remlen);
    for (int t = 0; t < rlh9::NSET; ++t)
        for (int S = 1; S <= rlh9::REMS; ++S)
            rlh9::huff_len(&rem_cnt[(size_t(t) * (rlh9::REMS + 1) + S) * (rlh9::REMS + 1)], S + 1, rlh9::REMB, remlen[t][S]);
    T0.build_rem(&remlen[0][0][0]);

    add_dir("../rlh_v2/data/data_photos_test", -1, 1.0, test_photos, nullptr);
    add_dir("../rlh_v2/data/test_full12", -1, 1.0, test_screens, nullptr);

    // Compute exact R8F4 baseline on same training corpus
    const auto& M = rlh9::fsm();
    std::vector<double> cnt4(size_t(rlh9::TOTC) * rlh9::NSYM, 0.0);
    for (const auto& im : train_imgs) for (const auto& ls : im.lanes) {
        uint32_t st = rlh9::ST0, cm = rlh9::ctx_m(ls.sc), cc = rlh9::ctx_c(ls.sc);
        const uint32_t gb = ls.sc * rlh9::SETC + ls.plane * rlh9::NCTX;
        for (const auto& t : ls.toks) {
            cnt4[size_t(gb + rlh9::ctx_of(st, cm, cc)) * rlh9::NSYM + t.sym] += im.wt;
            if (t.sym != rlh9::EOB) st = uint32_t(M.v[st & 63u][t.vh]) | (uint32_t(M.r[st >> 6][t.rc]) << 6);
        }
    }
    static uint8_t L4[rlh9::TOTC][rlh9::NSYM];
    for (int tc = 0; tc < rlh9::TOTC; ++tc) rlh9::huff_len(&cnt4[size_t(tc) * rlh9::NSYM], rlh9::NSYM, rlh9::MAXL, L4[tc]);
    auto eval_r8f4 = [&](const std::vector<ImgTokens>& imgs) {
        double tb = 0, tp = 0;
        for (const auto& im : imgs) {
            size_t bb = im.n_bands * 16;
            for (const auto& ls : im.lanes) {
                uint32_t st = rlh9::ST0, cm = rlh9::ctx_m(ls.sc), cc = rlh9::ctx_c(ls.sc);
                const uint32_t gb = ls.sc * rlh9::SETC + ls.plane * rlh9::NCTX;
                uint64_t lb = ls.fixed_bits;
                for (const auto& t : ls.toks) {
                    lb += L4[gb + rlh9::ctx_of(st, cm, cc)][t.sym];
                    if (t.sym != rlh9::EOB) st = uint32_t(M.v[st & 63u][t.vh]) | (uint32_t(M.r[st >> 6][t.rc]) << 6);
                }
                bb += size_t((lb + 7) >> 3);
            }
            tb += double(2 + (bb + 7) / 8) * 64.0; tp += double(im.W) * im.H;
        }
        return tb / tp;
    };
    const double base_p = eval_r8f4(test_photos), base_s = eval_r8f4(test_screens);
    std::printf("R8F4 BASELINE (72-img corpus): photos=%.4f bpp  screens=%.4f bpp\n\n", base_p, base_s);

    auto show = [&](const char* label, const CtxFsm& F) {
        double p = F.eval_bpp(test_photos), s = F.eval_bpp(test_screens);
        std::printf("%-36s | photos=%.4f (%+.3f%%) | screens=%.4f (%+.3f%%)\n",
                    label, p, (p - base_p) / base_p * 100.0, s, (s - base_s) / base_s * 100.0);
    };

    // Method 3: 48-state direct hysteresis
    auto F3 = build_method3_hyst(train_imgs);
    show("Method 3: 48-st direct hysteresis", F3);

    // Method 1: 48-state R8F4 majority quantization
    auto F1 = build_method1_quant(48, train_imgs);
    show("Method 1: 48-st R8F4 quant", F1);

    // Method 2: 48-state EM / k-means from Method 1
    CtxFsm F2 = F1;
    for (int it = 1; it <= 5; ++it) {
        F2.em_step(train_imgs, 25.0);
        char buf[64]; std::snprintf(buf, sizeof buf, "Method 2: 48-st EM iter %d", it);
        show(buf, F2);
    }

    // Q4: 64 states and 96 states (Method 1 + Method 2 EM)
    std::printf("\n=== Q4: Intermediate state counts (64 & 96 states, state == ctx) ===\n");
    for (int NC : {64, 96}) {
        auto FQ = build_method1_quant(NC, train_imgs);
        char b1[64]; std::snprintf(b1, sizeof b1, "Q4 %d-st Method 1 (R8F4 quant)", NC);
        show(b1, FQ);
        for (int it = 1; it <= 3; ++it) {
            FQ.em_step(train_imgs, 25.0);
            char b2[64]; std::snprintf(b2, sizeof b2, "Q4 %d-st Method 2 EM iter %d", NC, it);
            show(b2, FQ);
        }
    }

    // Export 48-state Method 2 EM transition table + Huffman lengths for Part B speed benchmark!
    FILE* hf = std::fopen("r8f5_best48_tables.h", "w");
    std::fprintf(hf, "#pragma once\n#include <cstdint>\n");
    std::fprintf(hf, "static const uint8_t R8F5_INIT_CTX[2] = {%d, %d};\n", F2.init_ctx[0], F2.init_ctx[1]);
    std::fprintf(hf, "static const uint8_t R8F5_TRANS[2][2][48][128] = {\n");
    for (int sc = 0; sc < 2; ++sc) {
        std::fprintf(hf, "{\n");
        for (int pl = 0; pl < 2; ++pl) {
            std::fprintf(hf, "{\n");
            for (int c = 0; c < 48; ++c) {
                std::fprintf(hf, "{");
                for (int cls = 0; cls < 128; ++cls) std::fprintf(hf, "%d%s", F2.trans(sc, pl, c, cls), cls < 127 ? "," : "");
                std::fprintf(hf, "}%s\n", c < 47 ? "," : "");
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
        const uint8_t* Lrow = F2.len_row(sc, pl, c);
        std::fprintf(hf, "{");
        for (int b = 0; b < PB; ++b) {
            int lo = Lrow[2 * b], hi = (2 * b + 1 < rlh9::NSYM) ? Lrow[2 * b + 1] : 0;
            std::fprintf(hf, "%d%s", lo | (hi << 4), b + 1 < PB ? "," : "");
        }
        std::fprintf(hf, "}%s\n", tc + 1 < rlh9::TOTC ? "," : "");
    }
    std::fprintf(hf, "};\n");
    std::fclose(hf);
    std::printf("\nExported r8f5_best48_tables.h!\n");
    return 0;
}
