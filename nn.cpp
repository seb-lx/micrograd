#include "nn.h"

#include <random>


namespace micrograd {


//
// Neuron
// 

Neuron::Neuron(std::size_t nin, bool nonlin):
    w_{},
    b_{ std::make_shared<Value>(0.0) },
    nonlin_{ nonlin }
{
    w_.reserve(nin);

    // Random weight init in [-1.0, 1.0]
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t i = 0; i < nin; ++i) {
        w_.emplace_back(std::make_shared<Value>(dist(gen)));
    }
}

auto Neuron::operator()(const std::vector<ValuePtr>& x) const -> ValuePtr {
    if (w_.size() != x.size()) {
        throw std::invalid_argument(
            std::format("Inputs need to be the same size as weights ({})!", w_.size())
        );
    }

    auto act = b_;
    for (std::size_t i = 0; i < w_.size(); ++i) {
        act = act + w_[i] * x[i]; 
    }

    auto out = nonlin_ ? tanh(act) : act;

    return out;
}

auto Neuron::parameters() const -> std::vector<ValuePtr> {
    auto params = std::vector<ValuePtr>{};
    params.reserve(w_.size() + 1);

    params.insert(params.end(), w_.begin(), w_.end());
    params.push_back(b_);

    return params;
}


//
// Layer
// 

Layer::Layer(std::size_t nin, std::size_t nout, bool nonlin):
    neurons_{} 
{
    neurons_.reserve(nout);

    for (std::size_t i = 0; i < nout; ++i) {
        neurons_.emplace_back(nin, nonlin);
    }
}

auto Layer::operator()(const std::vector<ValuePtr>& x) const -> std::vector<ValuePtr> {
    auto out = std::vector<ValuePtr>{};

    for (const auto& neuron: neurons_) {
        out.push_back(neuron(x));
    }

    return out;
}

auto Layer::parameters() const -> std::vector<ValuePtr> {
    auto params = std::vector<ValuePtr>{};

    for (const auto& neuron: neurons_) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end()); 
    }

    return params;
}


//
// MLP
// 

MLP::MLP(std::size_t nin, const std::vector<std::size_t>& nouts):
    layers_{} 
{
    layers_.reserve(nouts.size());

    auto sz = std::vector<std::size_t>{ nin };
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    for (std::size_t i = 0; i < nouts.size(); ++i) {
        bool is_output_layer = (i == nouts.size() - 1);
        layers_.emplace_back(sz[i], sz[i+1], !is_output_layer);
    }
}

auto MLP::operator()(const std::vector<ValuePtr>& x) const -> std::vector<ValuePtr> {
    auto out = x;
    for (const auto& layer: layers_) {
        out = layer(out);
    }

    return out;
}

auto MLP::parameters() const -> std::vector<ValuePtr> {
    auto params = std::vector<ValuePtr>{};

    for (const auto& layer: layers_) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());   
    }

    return params;
}


} // namespace micrograd
