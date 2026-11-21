// train_rlh7.cpp - trains ENGINE 7 tables (and optionally the regime selector) on the .rlv dumps.
//
// Differences from tools/rlh/train_rlh.cpp (ENGINE 6):
//   * the regime selector can be a learned 512-entry LUT over quantised block counters instead of
//     Argon's hand-tuned thresholds, with a configurable number of regimes;
//   * the smoothing strength K and the frequency SCALE are chosen PER REGIME on a held-out fold
//     instead of being hard-coded constants;
//   * the context alphabet can be reduced to the 14 coarse classes (smaller/faster tables).
//
//   train_rlh7 -o rlh7_tables.h -r rlh7_regime.h [-lut sel.bin] [-nregs N] [-ctx 34|14]
//              [-sc TARGET] [-ph W] [-tune] dir:sc|ph ...
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <filesystem>

static inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
static constexpr int NR = 19, NV = 44, NS = NR * NV, MAX_LEN = 12, PACK_BYTES = NS / 2;
static constexpr int NFOLD = 2, NGRP = 2;
static constexpr int MAXREG = 32;

static inline int vcls(uint32_t x) {
    const int b = bw(x);
    if (b <= 1) return b;
    if (b >= 3) return 4 * b - 4 + int((x >> (b - 3)) & 3);
    return 4 + int(x & 1);
}
static inline int next_ctx_of(int cr, int vk) {
    if (vk <= 0) return 0;
    return (std::min(vk, 11) - 1) * 3 + std::min(cr, 2) + 1;
}
static inline int c34_to_c14(int c34) {
    if (c34 == 0) return 0;
    int vk = (c34 - 1) / 3 + 1, cr = (c34 - 1) % 3;
    int kg = vk <= 3 ? vk : vk <= 5 ? 4 : vk <= 8 ? 5 : 6;
    return kg * 2 + (cr > 0);
}
static inline int get_reg_e6(uint32_t s, uint32_t h, uint32_t c) {
    if (s >= 20) { if (h >= 14) return 0; if (h >= 4 || s >= 96) return 1; return 2; }
    if (h >= 16 && s >= 6) return 0;
    if (c < 176) return 3; if (c < 266) return 4; if (c < 356) return 5; if (c < 452) return 6;
    return 7;
}
static inline int q_rl(uint32_t s) {
    if (s == 0) return 0; if (s < 6) return 1; if (s < 20) return 2; if (s < 48) return 3;
    if (s < 96) return 4; if (s < 192) return 5; if (s < 512) return 6; return 7;
}
static inline int q_hi(uint32_t h) {
    if (h < 4) return 0; if (h < 8) return 1; if (h < 14) return 2; if (h < 16) return 3;
    if (h < 20) return 4; if (h < 24) return 5; if (h < 28) return 6; return 7;
}
static inline int q_ctx(uint32_t c) {
    if (c < 176) return 0; if (c < 266) return 1; if (c < 356) return 2; if (c < 452) return 3;
    if (c < 600) return 4; if (c < 800) return 5; if (c < 1000) return 6; return 7;
}

// ---- canonical Huffman with a length limit (package-merge), same as ENGINE 6's trainer ----
static std::vector<uint8_t> build_huff_opt(std::vector<uint64_t> freq, int max_len) {
    int n = (int)freq.size();
    if (freq[0] == 0) freq[0] = 1;
    struct Item { uint64_t w; int sym; };
    std::vector<Item> items;
    for (int i = 0; i < n; ++i) if (freq[i] > 0) items.push_back({freq[i], i});
    int used = (int)items.size();
    std::vector<uint8_t> out_len(n, 0);
    if (used == 1) { out_len[items[0].sym] = 1; return out_len; }
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.w != b.w ? a.w < b.w : a.sym < b.sym;
    });
    struct Node { uint64_t w; std::vector<int> leaves; };
    std::vector<Node> base(used);
    for (int i = 0; i < used; ++i) base[i] = {items[i].w, {i}};
    std::vector<Node> cur;
    for (int d = 0; d < max_len; ++d) {
        std::vector<Node> pkgs;
        for (size_t i = 0; i + 1 < cur.size(); i += 2) {
            Node p; p.w = cur[i].w + cur[i + 1].w;
            p.leaves = cur[i].leaves;
            p.leaves.insert(p.leaves.end(), cur[i + 1].leaves.begin(), cur[i + 1].leaves.end());
            pkgs.push_back(std::move(p));
        }
        std::vector<Node> next_list;
        next_list.reserve(base.size() + pkgs.size());
        size_t ib = 0, ip = 0;
        while (ib < base.size() || ip < pkgs.size()) {
            if (ip >= pkgs.size() || (ib < base.size() && base[ib].w <= pkgs[ip].w))
                next_list.push_back(base[ib++]);
            else next_list.push_back(std::move(pkgs[ip++]));
        }
        cur = std::move(next_list);
    }
    std::vector<int> len(used, 0);
    for (int i = 0; i < 2 * used - 2 && i < (int)cur.size(); ++i)
        for (int lf : cur[i].leaves) len[lf]++;
    for (int i = 0; i < used; ++i) out_len[items[i].sym] = (uint8_t)len[i];
    return out_len;
}

// The decoder keeps codewords of length 11..12 in a 512-entry secondary table, 4 entries per
// distinct top-10 prefix.  A length set that needs more than 512 entries cannot be represented in
// the ENGINE 6/7 LUT format, so fall back to max_len 10 (everything fits the primary LUT).
static int sub_need(const std::vector<uint8_t>& len) {
    int bl[MAX_LEN + 1] = {};
    for (int s = 0; s < NS; ++s) if (len[s]) ++bl[len[s]];
    uint32_t nc[MAX_LEN + 1] = {}; uint32_t c = 0;
    for (int b = 1; b <= MAX_LEN; ++b) { c = (c + bl[b - 1]) << 1; nc[b] = c; }
    bool seen[1024] = {};
    int need = 0;
    for (int s = 0; s < NS; ++s) {
        const int l = len[s];
        if (!l) continue;
        const uint32_t cw = nc[l]++;
        if (l > 10) { const uint32_t t = cw >> (l - 10); if (!seen[t]) { seen[t] = true; need += 4; } }
    }
    return need;
}
static int g_capped = 0;
static std::vector<uint8_t> build_len(const std::vector<uint64_t>& f) {
    auto len = build_huff_opt(f, MAX_LEN);
    if (sub_need(len) > 512) { ++g_capped; len = build_huff_opt(f, 10); }
    return len;
}

struct Tok { uint32_t rl; uint16_t v; uint16_t pad; };

// hist[fold][grp][reg][ctx][sym]
struct Hists {
    int nreg, nctx;
    std::vector<uint64_t> h;
    Hists(int nr, int nc) : nreg(nr), nctx(nc), h(size_t(NFOLD) * NGRP * nr * nc * NS, 0) {}
    inline uint64_t& at(int f, int g, int r, int c, int s) {
        return h[(((size_t(f) * NGRP + g) * nreg + r) * nctx + c) * NS + s];
    }
    inline const uint64_t& at(int f, int g, int r, int c, int s) const {
        return h[(((size_t(f) * NGRP + g) * nreg + r) * nctx + c) * NS + s];
    }
};

int main(int argc, char** argv) {
    std::string out = "rlh7_tables.h", outr = "rlh7_regime.h", lutpath;
    int nregs = 8, nctx = 34, tune = 0;
    double sc_target_total = 3.6e6, ph_w = 2.0;
    std::vector<std::pair<std::string, bool>> dirs;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "-o") { out = argv[++i]; continue; }
        if (s == "-r") { outr = argv[++i]; continue; }
        if (s == "-lut") { lutpath = argv[++i]; continue; }
        if (s == "-nregs") { nregs = atoi(argv[++i]); continue; }
        if (s == "-ctx") { nctx = atoi(argv[++i]); continue; }
        if (s == "-sc") { sc_target_total = atof(argv[++i]); continue; }
        if (s == "-ph") { ph_w = atof(argv[++i]); continue; }
        if (s == "-tune") { tune = 1; continue; }
        const size_t c = s.rfind(':');
        dirs.push_back({s.substr(0, c), s.substr(c + 1) == "sc"});
    }
    if (nregs < 1 || nregs > MAXREG) { fprintf(stderr, "nregs 1..%d\n", MAXREG); return 1; }
    if (nctx != 34 && nctx != 14) { fprintf(stderr, "ctx must be 34 or 14\n"); return 1; }
    std::vector<uint8_t> lut(512, 0);
    bool use_lut = !lutpath.empty();
    if (use_lut) {
        FILE* f = fopen(lutpath.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", lutpath.c_str()); return 1; }
        if (fread(lut.data(), 1, 512, f) != 512) { fprintf(stderr, "bad lut\n"); return 1; }
        fclose(f);
        int mx = 0; for (int i = 0; i < 512; ++i) mx = std::max(mx, int(lut[i]));
        if (mx + 1 > nregs) { fprintf(stderr, "lut uses %d regimes, -nregs %d\n", mx + 1, nregs); return 1; }
    } else if (nregs != 8) { fprintf(stderr, "without -lut only 8 regimes (engine 6 selector)\n"); return 1; }

    Hists H(nregs, nctx);
    size_t tot = 0;
    for (auto& dp : dirs) {
        std::vector<std::string> files;
        for (auto& e : std::filesystem::directory_iterator(dp.first))
            if (e.path().extension() == ".rlv") files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        const double per_img = sc_target_total / double(files.size() ? files.size() : 1);
        for (auto& f : files) {
            FILE* g = fopen(f.c_str(), "rb");
            if (!g) continue;
            uint32_t hdr[4];
            if (fread(hdr, 4, 4, g) != 4 || hdr[0] != 0x31564c52u) { fclose(g); continue; }
            for (int ch = 0; ch < 3; ++ch) {
                uint64_t n = 0;
                if (fread(&n, 8, 1, g) != 1) break;
                std::vector<Tok> T(n);
                if (fread(T.data(), sizeof(Tok), n, g) != n) break;
                const int grp = (ch == 0) ? 0 : 1;
                const uint64_t w = dp.second ? uint64_t(std::max(1.0, per_img / double(n))) : uint64_t(ph_w);
                uint32_t sum_rl = 0, hi_vk = 0, sum_ctx = 0, tok_cnt = 0, blk = 0;
                int reg = 1, ctx = 0;
                for (uint64_t i = 0; i < n; ++i) {
                    const uint32_t rl = T[i].rl, v = T[i].v;
                    const int cr = bw(rl), vk = bw(v);
                    const bool esc = (rl >= (1u << 18)) || (v >= 2048u);
                    if (!esc && cr < NR) {
                        const int cv = vcls(v);
                        if (cv < NV) {
                            const int fold = int((blk >> 2) & 1u);
                            const int c = (nctx == 34) ? ctx : c34_to_c14(ctx);
                            H.at(fold, grp, reg, c, cr * NV + cv) += w;
                        }
                    }
                    ++tot;
                    if (i + 1 == n && v == 0) break;
                    const int nc = next_ctx_of(cr, vk);
                    sum_rl += std::min<uint32_t>(rl, 256u); hi_vk += (nc >= 16); sum_ctx += uint32_t(nc);
                    ctx = nc;
                    if ((++tok_cnt & 31u) == 0) {
                        ++blk;
                        reg = use_lut ? lut[(q_rl(sum_rl) << 6) | (q_hi(hi_vk) << 3) | q_ctx(sum_ctx)]
                                      : get_reg_e6(sum_rl, hi_vk, sum_ctx);
                        sum_rl = hi_vk = sum_ctx = 0;
                    }
                }
            }
            fclose(g);
        }
        fprintf(stderr, "  %s (%s): %zu files, tokens %zu\n", dp.first.c_str(), dp.second ? "sc" : "ph", files.size(), tot);
    }

    // ---- table construction ----
    const int cv_reach[2] = {36, 40};
    const int NC14 = (nctx == 34) ? 14 : nctx;
    std::vector<int> c14map(nctx);
    for (int c = 0; c < nctx; ++c) c14map[c] = (nctx == 34) ? c34_to_c14(c) : c;

    // ---- pre-aggregate everything that does not depend on (K, SCALE) ----
    // own[fold][ctx][sym] is H itself; build c14 / regime marginals and the family priors once.
    struct Agg {
        std::vector<double> c14[NFOLD], reg[NFOLD];
        std::vector<double> nc14[NFOLD], nreg[NFOLD], nown[NFOLD];
    };
    std::vector<std::vector<Agg>> AG(NGRP, std::vector<Agg>(nregs));
    std::vector<std::vector<char>> FAM(NGRP, std::vector<char>(nregs, 0));
    struct Prior { std::vector<double> cr, cv0, cv1; double t = 0, t0 = 0, t1 = 0; };
    std::vector<std::vector<Prior>> PRI(NGRP, std::vector<Prior>(nregs));   // [grp][reg]

    for (int grp = 0; grp < NGRP; ++grp) {
        for (int r = 0; r < nregs; ++r) {
            Agg& A = AG[grp][r];
            for (int f = 0; f < NFOLD; ++f) {
                A.c14[f].assign(size_t(NC14) * NS, 0.0);
                A.reg[f].assign(NS, 0.0);
                A.nc14[f].assign(NC14, 0.0);
                A.nreg[f].assign(1, 0.0);
                A.nown[f].assign(nctx, 0.0);
                for (int c = 0; c < nctx; ++c) {
                    const int k = c14map[c];
                    for (int s = 1; s < NS; ++s) {
                        const double v = double(H.at(f, grp, r, c, s));
                        if (v == 0) continue;
                        A.c14[f][size_t(k) * NS + s] += v; A.nc14[f][k] += v;
                        A.reg[f][s] += v; A.nreg[f][0] += v;
                        A.nown[f][c] += v;
                    }
                }
            }
            // family: does this regime carry runs? (share of tokens with cr > 0)
            double rl_w = 0, all = 0;
            for (int c = 0; c < nctx; ++c) for (int cr = 0; cr < NR; ++cr) for (int cv = 0; cv < NV; ++cv) {
                const double v = double(H.at(0, grp, r, c, cr * NV + cv)) + double(H.at(1, grp, r, c, cr * NV + cv));
                all += v; if (cr > 0) rl_w += v;
            }
            FAM[grp][r] = use_lut ? ((all > 0 && rl_w / all >= 0.30) ? 1 : 0) : (r <= 2 ? 1 : 0);
        }
        for (int r = 0; r < nregs; ++r) {
            Prior& P = PRI[grp][r];
            P.cr.assign(NR, 0.0); P.cv0.assign(NV, 0.0); P.cv1.assign(NV, 0.0);
            for (int rr = 0; rr < nregs; ++rr) {
                if (FAM[grp][rr] != FAM[grp][r]) continue;
                const double rw = (r == rr) ? 4.0 : 1.0;
                for (int c = 0; c < nctx; ++c) for (int cr = 0; cr < NR; ++cr) for (int cv = 0; cv < NV; ++cv) {
                    const double v = (double(H.at(0, grp, r, c, cr * NV + cv)) + double(H.at(1, grp, r, c, cr * NV + cv))) * rw;
                    P.cr[cr] += v; P.t += v;
                    if (cr == 0) { P.cv0[cv] += v; P.t0 += v; } else { P.cv1[cv] += v; P.t1 += v; }
                }
            }
        }
    }

    // frequency vector for one (grp,reg,ctx); fold_src < 0 = both folds
    auto freqs = [&](int grp, int r, int c, int fold_src, double K, double SCALE) {
        const Agg& A = AG[grp][r];
        const Prior& P = PRI[grp][r];
        const int k = c14map[c];
        const int reach = cv_reach[grp];
        double n_own = 0, n_c14 = 0, n_reg = 0;
        for (int rr = std::max(0, r - 1); rr <= std::min(nregs - 1, r + 1); ++rr) {
            const double rw = (rr == r) ? 1.0 : 0.25;
            const Agg& B = AG[grp][rr];
            for (int f = 0; f < NFOLD; ++f) {
                if (fold_src >= 0 && f != fold_src) continue;
                n_own += B.nown[f][c] * rw; n_c14 += B.nc14[f][k] * rw; n_reg += B.nreg[f][0] * rw;
            }
        }
        const double a_own = n_own / (n_own + K);
        const double a_c14 = (1.0 - a_own) * (n_c14 / (n_c14 + K * 0.5));
        const double a_reg = (1.0 - a_own - a_c14) * 0.80;
        const double a_fac = 1.0 - a_own - a_c14 - a_reg;
        std::vector<uint64_t> f(NS, 0);
        std::vector<double> po(NS, 0.0), pc(NS, 0.0), pr(NS, 0.0);
        for (int rr = std::max(0, r - 1); rr <= std::min(nregs - 1, r + 1); ++rr) {
            const double rw = (rr == r) ? 1.0 : 0.25;
            const Agg& B = AG[grp][rr];
            for (int fold = 0; fold < NFOLD; ++fold) {
                if (fold_src >= 0 && fold != fold_src) continue;
                for (int s = 1; s < NS; ++s) {
                    po[s] += double(H.at(fold, grp, rr, c, s)) * rw;
                    pc[s] += B.c14[fold][size_t(k) * NS + s] * rw;
                    pr[s] += B.reg[fold][s] * rw;
                }
            }
        }
        const double io = n_own > 0 ? 1.0 / n_own : 0.0;
        const double ic = n_c14 > 0 ? 1.0 / n_c14 : 0.0;
        const double ir = n_reg > 0 ? 1.0 / n_reg : 0.0;
        for (int cr = 0; cr < NR; ++cr) {
            const double pcr = (P.cr[cr] + 1.0) / (P.t + NR);
            for (int cv = 0; cv < reach; ++cv) {
                if (cr == 0 && cv == 0) continue;
                if (cr > 0 && cv == 0 && cr > 16) continue;
                const int s = cr * NV + cv;
                const double pcv = (cr == 0) ? (P.cv0[cv] + 1.0) / (P.t0 + reach) : (P.cv1[cv] + 1.0) / (P.t1 + reach);
                const double p = a_own * po[s] * io + a_c14 * pc[s] * ic + a_reg * pr[s] * ir + a_fac * pcr * pcv;
                const uint64_t w = uint64_t(p * SCALE);
                if (w > 0) f[s] = w; else f[0] += uint64_t(p * SCALE * 4.0 + 0.5);
            }
        }
        if (f[0] == 0) f[0] = 1;
        return f;
    };

    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> LEN(NGRP,
        std::vector<std::vector<std::vector<uint8_t>>>(nregs, std::vector<std::vector<uint8_t>>(nctx)));
    const double KS[] = {250.0, 1000.0, 4000.0, 12000.0, 48000.0};
    const double SCS[] = {3.5 * 4096, 12.0 * 4096, 24.0 * 4096, 96.0 * 4096, 384.0 * 4096};
    const int NK = 5, NSCv = 5;
    // mantissa bits per symbol, precomputed
    std::vector<int> MB(NS, 0);
    for (int s = 0; s < NS; ++s) {
        const int cr = s / NV, cv = s % NV;
        const int vk = cv == 0 ? 0 : (cv == 1 ? 1 : ((cv + 4) >> 2));
        MB[s] = (cr > 1 ? cr - 1 : 0) + (vk > 1 ? vk - 1 : 0);
    }
    for (int grp = 0; grp < NGRP; ++grp) {
        for (int r = 0; r < nregs; ++r) {
            double bestK = (r <= 2) ? 12000.0 : 1000.0;
            double bestS = 4096.0 * ((r <= 1) ? 96.0 : (r == 2 ? 24.0 : 3.5));
            if (tune) {
                double best = 1e300;
                for (int ik = 0; ik < NK; ++ik) for (int is = 0; is < NSCv; ++is) {
                    double cost = 0;
                    for (int lf = 0; lf < NFOLD; ++lf) {
                        const int ef = 1 - lf;
                        for (int c = 0; c < nctx; ++c) {
                            auto fq = freqs(grp, r, c, lf, KS[ik], SCS[is]);
                            auto len = build_len(fq);
                            const int esc = (len[0] ? len[0] : 12) + 9;
                            for (int s = 1; s < NS; ++s) {
                                const uint64_t n = H.at(ef, grp, r, c, s);
                                if (!n) continue;
                                cost += double(n) * (len[s] ? len[s] + MB[s] : esc + MB[s]);
                            }
                        }
                    }
                    if (cost < best) { best = cost; bestK = KS[ik]; bestS = SCS[is]; }
                }
                fprintf(stderr, "grp %d reg %2d: K=%.0f SCALE=%.0f (held-out %.3f Mbit)\n", grp, r, bestK, bestS / 4096.0, best / 1e6);
            }
            for (int c = 0; c < nctx; ++c) {
                auto fq = freqs(grp, r, c, -1, bestK, bestS);
                LEN[grp][r][c] = build_len(fq);
            }
        }
    }

    FILE* o = fopen(out.c_str(), "w");
    if (!o) return 1;
    fprintf(o, "// rlh7_tables.h - ENGINE 7 static canonical Huffman lengths (MAX_LEN 12, nibble packed).\n");
    fprintf(o, "// Trained by tools/rlh7/train_rlh7.cpp on icon-512 + icon-64 + screenshot-game + CLIC train photos.\n");
    fprintf(o, "// Quarter-octaves, NR=19, NV=44, NS=836. NREG=%d, NCTX=%d.\n", nregs, nctx);
    fprintf(o, "#pragma once\n#include <cstdint>\n\nnamespace rlh7 {\n");
    fprintf(o, "constexpr int NR = %d, NV = %d, NS = %d, NCTX = %d, NREG = %d, MAX_CLEN = %d, PACK_BYTES = %d;\n\n",
            NR, NV, NS, nctx, nregs, MAX_LEN, PACK_BYTES);
    auto write_arr = [&](const char* name, int grp) {
        fprintf(o, "alignas(64) static constexpr uint8_t %s[%d][%d][%d] = {\n", name, nregs, nctx, PACK_BYTES);
        for (int r = 0; r < nregs; ++r) {
            fprintf(o, "  {\n");
            for (int c = 0; c < nctx; ++c) {
                fprintf(o, "    {");
                for (int i = 0; i < NS; i += 2) {
                    uint8_t b = (LEN[grp][r][c][i] & 0x0Fu) | ((LEN[grp][r][c][i + 1] & 0x0Fu) << 4);
                    fprintf(o, "%u%s", unsigned(b), (i + 2 < NS) ? "," : "");
                }
                fprintf(o, "}%s\n", (c + 1 < nctx) ? "," : "");
            }
            fprintf(o, "  }%s\n", (r + 1 < nregs) ? "," : "");
        }
        fprintf(o, "};\n\n");
    };
    write_arr("RLH7_LEN_G", 0);
    write_arr("RLH7_LEN_C3", 1);
    fprintf(o, "} // namespace rlh7\n");
    fclose(o);

    FILE* q = fopen(outr.c_str(), "w");
    if (!q) return 1;
    fprintf(q, "// rlh7_regime.h - regime selector for ENGINE 7.\n");
    fprintf(q, "// %s\n", use_lut ? "learned 512-entry LUT over quantised block counters" : "ENGINE 6 hand-tuned thresholds");
    fprintf(q, "#pragma once\n#include <cstdint>\n\nnamespace rlh7 {\n");
    fprintf(q, "constexpr bool REG_LUT = %s;\n", use_lut ? "true" : "false");
    fprintf(q, "alignas(64) static constexpr uint8_t REG_SEL[512] = {");
    for (int i = 0; i < 512; ++i) fprintf(q, "%u%s", unsigned(lut[i]), i + 1 < 512 ? "," : "");
    fprintf(q, "};\n} // namespace rlh7\n");
    fclose(q);
    printf("wrote %s and %s (NREG=%d NCTX=%d, %zu tokens, %d tables capped to 10 bits)\n", out.c_str(), outr.c_str(), nregs, nctx, tot, g_capped);
    return 0;
}
