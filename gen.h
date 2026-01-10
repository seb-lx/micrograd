#pragma once

#include <vector>
#include <random>
#include <numbers>
#include <fstream>
#include <iostream>

#include "engine.h"


namespace micrograd {


struct Dataset {
    std::vector<std::vector<ValuePtr>> X;
    std::vector<double> y;
};


class DatasetGenerator {
public:
    DatasetGenerator(unsigned int seed = 42);

    auto make_moons(std::size_t n_samples, double noise) -> Dataset;

    static auto save_csv(const Dataset& ds, const std::string& filename) -> void;

private:
    std::mt19937 gen_;
};


} // namespace micrograd
