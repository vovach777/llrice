// train_rlh.cpp - trains static canonical Huffman code lengths (MAX_LEN = 12) for RLH (LLRICE_ENGINE 6)
// strictly on train datasets: dumps/screens_train, dumps/clic_train20 (and dumps_c2 for CODER 2 chroma rb).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>

static inline int bw(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
constexpr int NR = 19;      // cr = 0..18 in table (runs up to 262,143)
constexpr int NV = 44;      // cv = 0..43 in table (quarter-octaves HALF=2, vk up to 11)
constexpr int NS = NR * NV; // 836 symbols per table (sym = cr * NV + cv; sym 0 = SYM_ESC)
constexpr int NCTX = 34;    // 0: start/zero-v, 1..33: (min(vk, 11) - 1) * 3 + min(cr, 2) + 1
constexpr int NREG = 8;     // 0..2 screen regimes, 3..7 photo regimes
constexpr int MAX_LEN = 12; // max Huffman code length = 12 bits

struct Tok { int cr, cv; uint32_t rl, v; int vk; };

static inline int vcls(uint32_t x) {
    const int b = bw(x);
    if (b <= 1) return b;
    if (b >= 3) return 4 * b - 4 + int((x >> (b - 3)) & 3);
    return 4 * b - 4 + int((x >> (b - 2)) & 1);
}
static inline int mant_bits(int cr, int vk) {
    int rm = (cr > 1) ? (cr - 1) : 0;
    int vm = (vk > 1) ? (vk - 1 - (vk >= 2 ? 1 : 0) - (vk >= 3 ? 1 : 0)) : 0;
    return rm + vm;
}
static void tokenize(const std::vector<uint32_t>& s, std::vector<Tok>& t) {
    t.clear(); uint32_t rl = 0;
    for (uint32_t x : s) {
        if (!x) {
            if (++rl == (1u << 26) - 1u) { t.push_back({bw(rl), 0, rl, 0, 0}); rl = 0; }
            continue;
        }
        t.push_back({bw(rl), vcls(x), rl, x, bw(x)});
        rl = 0;
    }
    if (rl) t.push_back({bw(rl), 0, rl, 0, 0});
}
static inline int next_ctx_of(int cr, int vk) {
    if (vk <= 0) return 0;
    return (std::min(vk, 11) - 1) * 3 + std::min(cr, 2) + 1;
}
static inline int c34_to_c14(int c34) {
    if (c34 == 0) return 0;
    int vk = (c34 - 1) / 3 + 1;
    int cr = (c34 - 1) % 3;
    int kg = vk <= 3 ? vk : vk <= 5 ? 4 : vk <= 8 ? 5 : 6;
    return kg * 2 + (cr > 0);
}
static inline int get_reg(uint32_t sum_rl, uint32_t hi_vk, uint32_t sum_ctx) {
    if (sum_rl >= 20) {
        if (hi_vk >= 14) return 0;
        if (hi_vk >= 4 || sum_rl >= 96) return 1;
        return 2;
    }
    if (hi_vk >= 16 && sum_rl >= 6) return 0;
    if (sum_ctx < 176) return 3;
    if (sum_ctx < 266) return 4;
    if (sum_ctx < 356) return 5;
    if (sum_ctx < 452) return 6;
    return 7;
}

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
            else
                next_list.push_back(std::move(pkgs[ip++]));
        }
        cur = std::move(next_list);
    }
    std::vector<int> len(used, 0);
    for (int i = 0; i < 2 * used - 2 && i < (int)cur.size(); ++i) {
        for (int lf : cur[i].leaves) len[lf]++;
    }
    for (int i = 0; i < used; ++i) out_len[items[i].sym] = (uint8_t)len[i];
    return out_len;
}

struct Dataset {
    std::string name;
    double tot_llr_bits = 0, tot_px = 0;
    std::vector<std::vector<std::vector<Tok>>> imgs;
};

static Dataset load_ds(const std::string& dir, int ns) {
    Dataset ds; ds.name = dir;
    std::ifstream sz(dir + "/sizes.txt"); std::string line; std::vector<uint32_t> s;
    while (std::getline(sz, line)) {
        std::istringstream is(line); std::string nm; uint64_t bits; uint32_t w, h; int n;
        is >> nm >> bits >> w >> h >> n;
        ds.tot_llr_bits += double(bits) - 128.0; ds.tot_px += double(w) * h;
        std::vector<std::vector<Tok>> im(ns);
        for (int k = 0; k < ns; ++k) {
            std::ifstream f(dir + "/" + nm + ".s" + std::to_string(k), std::ios::binary | std::ios::ate);
            size_t L = f.tellg() / 4; f.seekg(0); s.resize(L); f.read((char*)s.data(), L * 4);
            tokenize(s, im[k]);
        }
        ds.imgs.push_back(std::move(im));
    }
    return ds;
}

int main() {
    Dataset c3_sc_tr = load_ds("dumps/screens_train", 3);
    Dataset c3_ph_tr = load_ds("dumps/clic_train20", 3);
    Dataset c2_sc_tr = load_ds("dumps_c2/screens_train", 2);
    Dataset c2_ph_tr = load_ds("dumps_c2/clic_train20", 2);

    std::vector<std::vector<std::vector<uint64_t>>> h_y(NREG, std::vector<std::vector<uint64_t>>(NCTX, std::vector<uint64_t>(NS, 0)));
    std::vector<std::vector<std::vector<uint64_t>>> h_c3(NREG, std::vector<std::vector<uint64_t>>(NCTX, std::vector<uint64_t>(NS, 0)));
    std::vector<std::vector<std::vector<uint64_t>>> h_c2(NREG, std::vector<std::vector<uint64_t>>(NCTX, std::vector<uint64_t>(NS, 0)));

    auto accum = [&](const Dataset& ds, bool is_sc, bool is_c2) {
        for (auto& im : ds.imgs) {
            if (is_sc && im[0].size() < 100) continue;
            uint64_t w_img = is_sc ? std::max<uint64_t>(1, 300000 / im[0].size()) : 2;
            for (int k = 0; k < (int)im.size(); ++k) {
                if (is_c2 && k == 0) continue;
                auto& h_all = (k == 0) ? h_y : (is_c2 ? h_c2 : h_c3);
                auto& t = im[k];
                uint32_t sum_rl = 0, hi_vk = 0, sum_ctx = 0, tok_cnt = 0;
                int reg = 1, ctx = 0;
                for (size_t i = 0; i < t.size(); ++i) {
                    if ((is_sc || reg >= 2) && t[i].cr < NR && t[i].cv < NV) {
                        h_all[reg][ctx][t[i].cr * NV + t[i].cv] += w_img;
                    }
                    ctx = next_ctx_of(t[i].cr, t[i].vk);
                    sum_rl += std::min<uint32_t>(t[i].rl, 256u);
                    hi_vk += (ctx >= 16);
                    sum_ctx += ctx;
                    if ((++tok_cnt & 31u) == 0) {
                        reg = get_reg(sum_rl, hi_vk, sum_ctx);
                        sum_rl = 0; hi_vk = 0; sum_ctx = 0;
                    }
                }
            }
        }
    };
    accum(c3_sc_tr, true, false);
    accum(c3_ph_tr, false, false);
    accum(c2_sc_tr, true, true);
    accum(c2_ph_tr, false, true);

    auto build_tables = [&](const auto& h_reg, int max_cv_reach) {
        std::vector<std::vector<std::vector<uint8_t>>> len_reg(NREG, std::vector<std::vector<uint8_t>>(NCTX));
        std::vector<std::vector<std::vector<double>>> h14(NREG, std::vector<std::vector<double>>(14, std::vector<double>(NS, 0.0)));
        std::vector<std::vector<double>> h_r(NREG, std::vector<double>(NS, 0.0));
        std::vector<std::vector<double>> r_cr(NREG, std::vector<double>(NR, 0.0));
        std::vector<std::vector<double>> r_cv0(NREG, std::vector<double>(NV, 0.0));
        std::vector<std::vector<double>> r_cv1(NREG, std::vector<double>(NV, 0.0));

        for (int r = 0; r < NREG; ++r) {
            for (int c = 0; c < NCTX; ++c) {
                int c14 = c34_to_c14(c);
                for (int s = 1; s < NS; ++s) {
                    double v = h_reg[r][c][s];
                    h14[r][c14][s] += v;
                    h_r[r][s] += v;
                    int cr = s / NV, cv = s % NV;
                    r_cr[r][cr] += v;
                    if (cr == 0) r_cv0[r][cv] += v;
                    else r_cv1[r][cv] += v;
                }
            }
        }

        for (int r = 0; r < NREG; ++r) {
            int r_lo = (r <= 2) ? 0 : 3;
            int r_hi = (r <= 2) ? 2 : 7;
            std::vector<double> dom_cr(NR, 0.0), dom_cv0(NV, 0.0), dom_cv1(NV, 0.0);
            double dom_tot = 0, dom_tot0 = 0, dom_tot1 = 0;
            for (int r2 = r_lo; r2 <= r_hi; ++r2) {
                double rw = (r2 == r) ? 4.0 : 1.0;
                for (int cr = 0; cr < NR; ++cr) { dom_cr[cr] += r_cr[r2][cr] * rw; dom_tot += r_cr[r2][cr] * rw; }
                for (int cv = 0; cv < NV; ++cv) {
                    dom_cv0[cv] += r_cv0[r2][cv] * rw; dom_tot0 += r_cv0[r2][cv] * rw;
                    dom_cv1[cv] += r_cv1[r2][cv] * rw; dom_tot1 += r_cv1[r2][cv] * rw;
                }
            }

            for (int c = 0; c < NCTX; ++c) {
                int c14 = c34_to_c14(c);
                double sum_rc = 0, sum_rc14 = 0, sum_r = 0;
                std::vector<double> emp_rc(NS, 0.0), emp_rc14(NS, 0.0), emp_r(NS, 0.0);
                for (int r2 = std::max(r_lo, r - 1); r2 <= std::min(r_hi, r + 1); ++r2) {
                    double rw = (r2 == r) ? 1.0 : 0.25;
                    for (int s = 1; s < NS; ++s) {
                        emp_rc[s]   += h_reg[r2][c][s] * rw;    sum_rc   += h_reg[r2][c][s] * rw;
                        emp_rc14[s] += h14[r2][c14][s] * rw;    sum_rc14 += h14[r2][c14][s] * rw;
                        emp_r[s]    += h_r[r2][s] * rw;         sum_r    += h_r[r2][s] * rw;
                    }
                }
                double k_prior = (r <= 2) ? 12000.0 : 1000.0;
                double alpha_c34 = sum_rc / (sum_rc + k_prior);
                double alpha_c14 = (1.0 - alpha_c34) * (sum_rc14 / (sum_rc14 + k_prior * 0.5));
                double alpha_reg = (1.0 - alpha_c34 - alpha_c14) * 0.80;
                double alpha_fac = 1.0 - alpha_c34 - alpha_c14 - alpha_reg;

                std::vector<uint64_t> f(NS, 0);
                const double SCALE = double(1ull << MAX_LEN) * ((r <= 1) ? 96.0 : (r == 2 ? 24.0 : 3.5));
                for (int cr = 0; cr < NR; ++cr) {
                    for (int cv = 0; cv < max_cv_reach; ++cv) {
                        if (cr == 0 && cv == 0) continue;
                        if (cr > 0 && cv == 0 && cr > 16) continue;
                        int s = cr * NV + cv;
                        double p34  = sum_rc > 0   ? (emp_rc[s] / sum_rc)     : 0.0;
                        double p14  = sum_rc14 > 0 ? (emp_rc14[s] / sum_rc14) : 0.0;
                        double preg = sum_r > 0    ? (emp_r[s] / sum_r)       : 0.0;
                        double p_cr = (dom_cr[cr] + 1.0) / (dom_tot + NR);
                        double p_cv = (cr == 0)
                            ? ((dom_cv0[cv] + 1.0) / (dom_tot0 + max_cv_reach))
                            : ((dom_cv1[cv] + 1.0) / (dom_tot1 + max_cv_reach));
                        double pfac = p_cr * p_cv;
                        double p = alpha_c34 * p34 + alpha_c14 * p14 + alpha_reg * preg + alpha_fac * pfac;
                        uint64_t w = uint64_t(p * SCALE);
                        if (w > 0) f[s] = w;
                        else f[0] += uint64_t(p * SCALE * 4.0 + 0.5);
                    }
                }
                if (f[0] == 0) f[0] = 1;
                len_reg[r][c] = build_huff_opt(f, MAX_LEN);
            }
        }
        return len_reg;
    };

    auto len_y  = build_tables(h_y, 36);
    auto len_c3 = build_tables(h_c3, 40);
    auto len_c2 = build_tables(h_c2, 44);

    // Emit llrice/rlh_tables.h using compact nibble packing (418 bytes per table: 2 symbols per byte)
    FILE* out = fopen("llrice/rlh_tables.h", "w");
    fprintf(out, "// rlh_tables.h - Static canonical Huffman code lengths for RLH (LLRICE_ENGINE 6)\n");
    fprintf(out, "// Trained strictly on data_screens_train_p6 (12 screens) and clic/p6 (first 20 photos).\n");
    fprintf(out, "// Quarter-octaves (HALF = 2), NR = 19 (cr = 0..18), NV = 44 (cv = 0..43), NS = 836, MAX_LEN = 12.\n");
    fprintf(out, "// Packed 2 nibbles per byte (low nibble = even symbol, high nibble = odd symbol): 418 bytes/table.\n");
    fprintf(out, "#pragma once\n#include <cstdint>\n\nnamespace rlh {\n");
    fprintf(out, "constexpr int NR = %d, NV = %d, NS = %d, NCTX = %d, NREG = %d, MAX_CLEN = %d, PACK_BYTES = %d;\n\n",
            NR, NV, NS, NCTX, NREG, MAX_LEN, NS / 2);

    auto write_arr = [&](const char* name, const auto& tbl) {
        fprintf(out, "alignas(64) static constexpr uint8_t %s[%d][%d][%d] = {\n", name, NREG, NCTX, NS / 2);
        for (int r = 0; r < NREG; ++r) {
            fprintf(out, "  {\n");
            for (int c = 0; c < NCTX; ++c) {
                fprintf(out, "    {");
                for (int i = 0; i < NS; i += 2) {
                    uint8_t b = (tbl[r][c][i] & 0x0Fu) | ((tbl[r][c][i + 1] & 0x0Fu) << 4);
                    fprintf(out, "%u%s", unsigned(b), (i + 2 < NS) ? "," : "");
                }
                fprintf(out, "}%s\n", (c + 1 < NCTX) ? "," : "");
            }
            fprintf(out, "  }%s\n", (r + 1 < NREG) ? "," : "");
        }
        fprintf(out, "};\n\n");
    };

    write_arr("RLH_LEN_G", len_y);
    write_arr("RLH_LEN_C3", len_c3);
    write_arr("RLH_LEN_C2", len_c2);
    fprintf(out, "} // namespace rlh\n");
    fclose(out);
    printf("Generated llrice/rlh_tables.h successfully!\n");
}
