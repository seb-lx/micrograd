# micrograd
Implementation of micrograd in modern C++ from the course "Neural Networks: Zero to Hero" by Andrej Karpathy (https://karpathy.ai/zero-to-hero.html).

### results
![generated moons dataset](plot/moons_dataset.csv)

![decision boundary](plot/moons_decision_boundary.csv)

### build
debug

g++ -std=c++20 -pedantic-errors -ggdb -Wall -Weffc++ -Wextra -Wconversion -Wsign-conversion -Werror engine.cpp nn.cpp gen.cpp main.cpp -o main

release

g++ -std=c++20 -pedantic-errors -O2 -DNDEBUG engine.cpp nn.cpp gen.cpp main.cpp -o main
