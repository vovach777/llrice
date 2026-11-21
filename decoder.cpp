#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include "llrice.hpp"
#include "profiling.hpp"

int main(int argc, char** argv) {
    profiling::StopWatch execute_time;
    if (argc < 3) {
        std::cout << "LLR - Simple solution good compression" << std::endl;
        std::cout << "LLRice is streming frendly intraframe lossless image compressor." << std::endl;
        std::cout << "Only LLR file type is supported for input!" << std::endl;
        std::cout << "Please use this format: \"" << "llrice-d" << " input_file output_file\"" << std::endl;
        return 0;
    }
    const char* filename = argv[1];
    const char* outputFilename = argv[2];
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return 1;
    }
    std::string line;
    if (!std::getline(inFile, line) || line.size() > 256) { std::cerr << "Not an llrice file: " << filename << std::endl; return 1; }
    const std::string tag = llrice::build_tag();
    if (line != tag) {
        std::cerr << "The file was made by \"" << line << "\", this decoder is \"" << tag << "\"" << std::endl;
        return 1;
    }
    const std::streamoff body = inFile.tellg();
    const size_t size = size_t(inFile.seekg(0, std::ios::end).tellg() - body) / sizeof(uint64_t);
    inFile.seekg(body, std::ios::beg);
    std::vector<uint64_t> compressed(size);
    inFile.read(reinterpret_cast<char*>(compressed.data()), std::streamsize(size * sizeof(uint64_t)));
    inFile.close();
    try {
        execute_time.startnew();
        auto rawImage = llrice::decompress(compressed);
        execute_time.stop();
        rawImage.save(std::string(outputFilename));

    } catch (const std::exception& e) {
        std::cerr << "Error decompressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }

    //std::cout << "time: " << execute_time.elapsed_str() << std::endl;
    return 0;
}
