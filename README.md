# micrograd
Implementation of Andrej Karpathy's micrograd in C++. Following the video series "Neural Networks: Zero to Hero" by Andrej Karpathy (https://karpathy.ai/zero-to-hero.html).

The core of this engine consists of the Value class which is a wrapper around floating point numbers. It also stores the gradient which is set during the backpropagation algorithm and computed using the chain rule of derivatives.

The network can be trained by iteratively updating the weights based on their individual influence on the overall loss function by following the direction of the gradients in small steps.

### results
The neural network is tested by predicting a decision boundary to separate moon-shaped data points.

<img src="https://github.com/seb-lx/micrograd/blob/main/plot/moons_dataset.png" alt="Alt text" width="700">
<img src="https://github.com/seb-lx/micrograd/blob/main/plot/decision_boundary.png" alt="Alt text" width="700">

#### build debug
g++ -std=c++20 -pedantic-errors -ggdb -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror engine.cpp nn.cpp gen.cpp main.cpp -o main

#### build release
g++ -std=c++20 -pedantic-errors -O2 -DNDEBUG engine.cpp nn.cpp gen.cpp main.cpp -o main
