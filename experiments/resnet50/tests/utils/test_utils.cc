#include "test_utils.h"
#include "op/utils.h"
#include <fstream>
#include <iostream>
#include <random>

namespace mica::experiments::models {
std::vector<float> ReadAll(const std::string &file, const std::string &dir) {
    std::vector<float> result;
    std::ifstream is(dir + '/' + file, std::ifstream::binary);
    char buf[4096];
    ASSERT(is && !is.bad() && !is.eof());

    while (!is.bad() && !is.eof()) {
        is.read(buf, sizeof(buf));
        size_t count = is.gcount();
        if (!count) {
            break;
        }
        ASSERT(!is.bad() && count % sizeof(float) == 0);
        result.insert(result.end(), reinterpret_cast<float *>(buf),
                      reinterpret_cast<float *>(buf + count));
    }
    return result;
}

std::vector<float> RandomData(size_t size) {
    std::vector<float> data(size);
    enum {
        kRandomSeed = 0xdeadbeef,
    };
    std::default_random_engine random(kRandomSeed);
    std::uniform_real_distribution<float> dist(0.0, 0.001);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(random);
    }
    return data;
}
} // namespace mica::experiments::models