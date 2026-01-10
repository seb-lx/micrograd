#include "gen.h"


namespace micrograd {


DatasetGenerator::DatasetGenerator(unsigned int seed): gen_{ seed } {}

auto DatasetGenerator::make_moons(std::size_t n_samples, double noise) -> Dataset {
    auto ds = Dataset{};

    std::normal_distribution<double> distr_noise(0.0, noise);

    std::size_t n_samples_per_moon = n_samples / 2;

    // First moon
    for (std::size_t i = 0; i < n_samples_per_moon; ++i) {
        auto angle = std::numbers::pi * static_cast<double>(i) / static_cast<double>(n_samples_per_moon);
        auto x = std::cos(angle) + distr_noise(gen_);
        auto y = std::sin(angle) + distr_noise(gen_);
        ds.X.push_back({
            std::make_shared<Value>(x),
            std::make_shared<Value>(y)
        });
        ds.y.push_back(1.0);
    }

    // Second moon
    for (std::size_t i = 0; i < n_samples_per_moon; ++i) {
        auto angle = std::numbers::pi * static_cast<double>(i) / static_cast<double>(n_samples_per_moon);
        auto x = 1.0 - std::cos(angle) + distr_noise(gen_);
        auto y = 0.5 - std::sin(angle) + distr_noise(gen_);
        ds.X.push_back({
            std::make_shared<Value>(x),
            std::make_shared<Value>(y)
        });
        ds.y.push_back(-1.0);
    }

    return ds;
}


auto DatasetGenerator::save_csv(const Dataset& ds, const std::string& filename) -> void {
    std::ofstream file(filename);
    std::cout << "[micrograd::DatasetGenerator::save_csv] Saving dataset to file " << filename << "\n";
    file << "x,y,label\n";
    for (size_t i = 0; i < ds.X.size(); ++i) {
        file << ds.X[i][0]->data << "," << ds.X[i][1]->data << "," << ds.y[i] << "\n";
    }
    file.close();
}


} // namespace micrograd
