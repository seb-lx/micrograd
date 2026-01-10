# micrograd
Implementation of Andrej Karpathy's micrograd in C++. Following the video series "Neural Networks: Zero to Hero" by Andrej Karpathy (https://karpathy.ai/zero-to-hero.html).

### results
The neural network is used to predict a decision boundary to separate moon-shaped data points.

<img src="https://github.com/seb-lx/micrograd/blob/main/plot/moons_dataset.png" alt="Alt text" width="700">
<img src="https://github.com/seb-lx/micrograd/blob/main/plot/decision_boundary.png" alt="Alt text" width="700">

### build debug
g++ -std=c++20 -pedantic-errors -ggdb -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror engine.cpp nn.cpp gen.cpp main.cpp -o main

### build release
g++ -std=c++20 -pedantic-errors -O2 -DNDEBUG engine.cpp nn.cpp gen.cpp main.cpp -o main
