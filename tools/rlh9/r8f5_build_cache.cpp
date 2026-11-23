// Build binary token cache for all images used by the screen-half trainer.
// Runs extract_image() once per image; trainer then reads tokens from disk.
#include "r8f5_common.hpp"

int main(int argc, char** argv) {
    const char* BASE = "/tmp/r8f5/scr";
    const char* CACHE = "/tmp/r8f5/cache";
    const rlh9::Tables& T0 = rlh9::tables();
    std::vector<std::string> dirs;
    for (int i = 1; i < argc; ++i) dirs.push_back(argv[i]);
    if (dirs.empty()) {
        dirs = { std::string(BASE)+"/ios/all", std::string(BASE)+"/desktop/all",
                 std::string(BASE)+"/websight/all", std::string(BASE)+"/docs/all",
                 std::string(BASE)+"/scans/all", std::string(BASE)+"/web/all",
                 std::string(BASE)+"/full12", std::string(BASE)+"/web8",
                 std::string(BASE)+"/valid", std::string(BASE)+"/kodak" };
    }
    auto t0 = std::chrono::steady_clock::now();
    size_t grand = 0;
    for (auto& d : dirs) {
        // cache path mirrors the directory path under CACHE
        std::string cdir = std::string(CACHE) + d.substr(std::strlen(BASE));
        std::filesystem::create_directories(cdir);
        auto files = list_ppms(d);
        size_t ntok = 0;
        for (auto& f : files) {
            std::string name = std::filesystem::path(f).filename().string();
            std::string cp = cdir + "/" + std::filesystem::path(name).replace_extension(".tok").string();
            ImgTokens im;
            extract_image(f, -1, 1.0, T0, im);   // detector; wt set later by trainer
            write_cache(cp, im);
            for (auto& ls : im.lanes) ntok += ls.toks.size();
        }
        grand += ntok;
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::printf("[%6.1fs] %-28s %5zu imgs  %12zu tokens\n", sec, d.c_str(), files.size(), ntok);
        std::fflush(stdout);
    }
    std::printf("TOTAL tokens=%zu\n", grand);
    return 0;
}
