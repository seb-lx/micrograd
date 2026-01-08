#pragma once

#include "engine.h"


namespace micrograd {


class Module {
public:
    virtual ~Module() = default;

    virtual auto parameters() const -> std::vector<ValuePtr> = 0;

    auto zero_grad() -> void {
        for (const auto& param: parameters()) param->grad = 0.0;
    }
};


class Neuron: public Module {
public:
    explicit Neuron(std::size_t nin, bool nonlin = true);

    auto operator()(const std::vector<ValuePtr>& x) const -> ValuePtr;

    auto parameters() const -> std::vector<ValuePtr>;

    friend auto operator<<(std::ostream& stream, const Neuron& neuron) -> std::ostream& {
        stream << std::format("{} Neuron({})", (neuron.nonlin_ ? "tanh" : "linear"), neuron.w_.size()); 
        return stream;
    }

private:
    std::vector<ValuePtr> w_;
    ValuePtr b_;
    bool nonlin_;
};


class Layer: public Module {
public:
    Layer(std::size_t nin, std::size_t nout, bool nonlin = true);

    auto operator()(const std::vector<ValuePtr>& x) const -> std::vector<ValuePtr>;

    auto parameters() const -> std::vector<ValuePtr>;

    friend auto operator<<(std::ostream& stream, const Layer& layer) -> std::ostream& {
        stream << "Layer of [ ";

        for (const auto& neuron: layer.neurons_) {
            stream << neuron << " ";
        }

        stream << "]";

        return stream;
    }

private:
    std::vector<Neuron> neurons_;
};


class MLP: public Module {
public:
    MLP(std::size_t nin, const std::vector<std::size_t>& nouts);

    auto operator()(const std::vector<ValuePtr>& x) const -> std::vector<ValuePtr>;

    auto parameters() const -> std::vector<ValuePtr>;

    friend auto operator<<(std::ostream& stream, const MLP& mlp) -> std::ostream& {
        stream << "MLP of [ ";

        for (const auto& layer: mlp.layers_) {
            stream << layer << " ";
        }

        stream << "]";

        return stream;
    }

private:
    std::vector<Layer> layers_;
};


} // namespace micrograd
