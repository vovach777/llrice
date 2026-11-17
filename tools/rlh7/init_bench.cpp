// init_bench.cpp - cost of building the static table working set (LUTs) at startup.
#include <chrono>
#include <cstdio>
#include <memory>
#include "llrice.hpp"
#if LLRICE_ENGINE == 6 || LLRICE_ENGINE == 7 || LLRICE_ENGINE == 8
int main() {
    using clk = std::chrono::steady_clock;
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        auto t0 = clk::now();
        std::unique_ptr<llrice::fast_ns::GlobalTables> g(new llrice::fast_ns::GlobalTables());
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
        if (g->enc[0][0][0].len[1] == 255) printf("x");
    }
    printf("sizeof(GlobalTables) %zu B  init %.3f ms\n", sizeof(llrice::fast_ns::GlobalTables), best * 1e3);
    return 0;
}
#else
int main() { printf("no static tables (adaptive RLGR): init 0 ms\n"); return 0; }
#endif
