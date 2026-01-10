# micrograd
Implementation of micrograd in modern C++ from the course "Neural Networks: Zero to Hero" by Andrej Karpathy (https://karpathy.ai/zero-to-hero.html).

### results
<img src="https://github.com/seb-lx/micrograd/blob/main/plot/moons_dataset.png" alt="Alt text" width="800">
<img src="https://github.com/seb-lx/micrograd/blob/main/plot/decision_boundary.png" alt="Alt text" width="800">

### build
debug

g++ -std=c++20 -pedantic-errors -ggdb -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror engine.cpp nn.cpp gen.cpp main.cpp -o main

release

g++ -std=c++20 -pedantic-errors -O2 -DNDEBUG engine.cpp nn.cpp gen.cpp main.cpp -o main
