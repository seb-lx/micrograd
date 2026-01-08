#include "engine.h"
#include "nn.h"


auto main() -> int {
    using micrograd::MLP;
    using micrograd::Value;
    using micrograd::ValuePtr;

    auto nn = MLP(3, { 4, 4, 1 });

    auto xs = std::vector<std::vector<ValuePtr>>{
        { std::make_shared<Value>(2.0), std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0) },
        { std::make_shared<Value>(3.0), std::make_shared<Value>(-1.0), std::make_shared<Value>(0.5) },
        { std::make_shared<Value>(0.5), std::make_shared<Value>(1.0), std::make_shared<Value>(1.0) },
        { std::make_shared<Value>(1.0), std::make_shared<Value>(1.0), std::make_shared<Value>(-1.0) }
    };

    auto ys = std::vector<double>{ 1.0, -1.0, -1.0, 1.0 };

    unsigned int iterations = 25;
    double learning_rate = 0.05;

    for (unsigned int i = 0; i < iterations; ++i) {

        // Forward pass
        auto ypred = std::vector<ValuePtr>{};
        for (const auto& x: xs) ypred.push_back(nn(x)[0]);

        auto loss = std::make_shared<Value>(0.0);
        for (std::size_t i = 0; i < ypred.size(); ++i) {
            loss = loss + pow(ypred[i] - ys[i], 2.0);
        }

        // Backward pass
        nn.zero_grad();
        backward(loss);

        // Update
        for (const auto& param: nn.parameters()) {
            param->data -= learning_rate * param->grad;
        }
        
        // Print progress
        std::cout << "Iteration " << i << ", loss = " << loss->data << "\n";
    }


    return 0;
}
