#include <tuple>

#include "engine.h"
#include "nn.h"
#include "gen.h"


auto test_simple_example() -> void;
auto test_moons_dataset() -> void;

auto loss_f(
    const micrograd::MLP& model,
    const std::vector<std::vector<micrograd::ValuePtr>>& X,
    const std::vector<double>& y
) -> std::pair<micrograd::ValuePtr, double>;

auto save_decision_boundary(
    const micrograd::MLP& model,
    const std::vector<std::vector<micrograd::ValuePtr>>& X,
    const std::vector<double>& y
) -> void;


auto main() -> int {

    //test_simple_example();
    test_moons_dataset();

    return 0;
}

auto test_simple_example() -> void {
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
}


auto test_moons_dataset() -> void {
    auto ds_gen = micrograd::DatasetGenerator();
    auto moons = ds_gen.make_moons(100, 0.1);

    //micrograd::DatasetGenerator::save_csv(moons, "moons.csv");

    auto& X = moons.X;
    auto& y = moons.y;

    auto model = micrograd::MLP(2, { 16, 16, 1 });
    std::cout << "Model (with " << model.parameters().size() << " parameters):\n" << model << "\n";

    // Training
    std::size_t iterations = 100;
    for (std::size_t k = 0; k < iterations; ++k) {
        // Forward pass
        auto res = loss_f(model, X, y);
        auto& total_loss = res.first;
        auto& acc = res.second;

        // Backward pass
        model.zero_grad();
        backward(total_loss);

        // Update
        double learning_rate = 1.0 - (0.9 * static_cast<double>(k) / 100);
        for (const auto& param: model.parameters()) {
            param->data -= learning_rate * param->grad;
        }
        
        // Print progress
        if (k % 1 == 0) {
            std::cout << "Iteration " << k << ", loss = " << total_loss->data << ", acc = " << acc * 100 << "%\n";
        }
    }

    //save_decision_boundary(model, X, y);
}

auto loss_f(
    const micrograd::MLP& model,
    const std::vector<std::vector<micrograd::ValuePtr>>& X,
    const std::vector<double>& y
) -> std::pair<micrograd::ValuePtr, double>
{
    std::vector<micrograd::ValuePtr> scores;
    scores.reserve(X.size());
    
    for (const auto& sample : X) scores.push_back(model(sample)[0]);
        
    auto data_loss = std::make_shared<micrograd::Value>(0.0);
    for (std::size_t i = 0; i < y.size(); ++i) {
        auto margin = 1.0 + ((-1.0 * y[i]) * scores[i]);
        data_loss = data_loss + micrograd::relu(margin);
    }

    data_loss = data_loss * (1.0 / static_cast<double>(y.size()));

    double alpha = 1e-4;
    auto reg_loss = std::make_shared<micrograd::Value>(0.0);
    
    for (const auto& p : model.parameters()) reg_loss = reg_loss + (p * p);
    reg_loss = reg_loss * alpha;

    auto total_loss = data_loss + reg_loss;

    // accuracy
    std::size_t correct_count = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        bool pred_positive = scores[i]->data > 0;
        bool truth_positive = y[i] > 0;
        
        if (pred_positive == truth_positive) correct_count++;
    }
    
    double accuracy = static_cast<double>(correct_count) / static_cast<double>(y.size());

    return { total_loss, accuracy };
}

auto save_decision_boundary(
    const micrograd::MLP& model,
    const std::vector<std::vector<micrograd::ValuePtr>>& X,
    const std::vector<double>& y
) -> void
{
    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();

    for (const auto& sample: X) {
        double val_x = sample[0]->data;
        double val_y = sample[1]->data;
        if (val_x < x_min) x_min = val_x;
        if (val_x > x_max) x_max = val_x;
        if (val_y < y_min) y_min = val_y;
        if (val_y > y_max) y_max = val_y;
    }

    double padding = 1.0;
    x_min -= padding; x_max += padding;
    y_min -= padding; y_max += padding;

    std::ofstream grid_file("moons_decision_boundary.csv");
    grid_file << "x,y,score\n";

    double h = 0.1;
    for (double xx = x_min; xx <= x_max; xx += h) {
        for (double yy = y_min; yy <= y_max; yy += h) {
            
            std::vector<micrograd::ValuePtr> inputs = {
                std::make_shared<micrograd::Value>(xx),
                std::make_shared<micrograd::Value>(yy)
            };

            auto score = model(inputs)[0];

            grid_file << xx << "," << yy << "," << score->data << "\n";
        }
    }
    
    grid_file.close();
    std::cout << "saved decision boundary\n";
}
