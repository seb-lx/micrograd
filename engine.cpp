#include "engine.h"

#include <cmath>


namespace micrograd {


auto Value::print_graph(std::size_t depth) const -> void {
    std::string indent(depth * 4, ' ');
    std::cout << indent << std::format("Value({:.2f}) op='{}' grad='{:.2f}'\n", data, op, grad);
    for (const auto& child: children) {
        child->print_graph(depth + 1);
    }
}


auto backward(const ValuePtr& root) -> void {
    std::vector<ValuePtr> topo;
    std::unordered_set<Value*> visited;

    auto build_topo = [&](auto&& self, ValuePtr v) -> void {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (const auto& child : v->children) {
                self(self, child);
            }
            topo.push_back(v);
        }
    };

    build_topo(build_topo, root);

    root->grad = 1.0;
    std::ranges::reverse(topo);
    for (const auto& v: topo) {
        v->backward();
    }
}


auto operator+(const ValuePtr& left, const ValuePtr& right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left->data + right->data,
        std::vector{ left, right },
        "+"
    );

    // Use raw pointer here since capturing smart pointer creates cycle
    // Safe, since lambda lives in object out which is thus guaranteed
    // to be alive when out->backward() is called.
    auto const* out_ptr = out.get();
    out->backward = [left, right, out_ptr]() {
        left->grad += out_ptr->grad;
        right->grad += out_ptr->grad;
    };

    return out;
}

// Addition overload for constant left parameter
auto operator+(double left, const ValuePtr& right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left + right->data,
        std::vector{ right },
        "+" //std::format("+ {:.2f}", left) // add constant to operation string so that it is visible in computational graph
    );

    auto const* out_ptr = out.get();
    out->backward = [right, out_ptr]() {
        right->grad += out_ptr->grad;
    };

    return out;
}

// Addition overload for constant right parameter
auto operator+(const ValuePtr& left, double right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left->data + right,
        std::vector{ left },
        "+" //std::format("+ {:.2f}", right)
    );

    auto const* out_ptr = out.get();
    out->backward = [left, out_ptr]() {
        left->grad += out_ptr->grad;
    };

    return out;
}


auto operator-(const ValuePtr& left, const ValuePtr& right) -> ValuePtr {
    return left + (right * -1.0);
}

// Subtraction overload for constant left parameter
auto operator-(double left, const ValuePtr& right) -> ValuePtr {
    return left + (right * -1.0);
}

// Subtraction overload for constant right parameter
auto operator-(const ValuePtr& left, double right) -> ValuePtr {
    return left + (-right);
}


auto operator*(const ValuePtr& left, const ValuePtr& right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left->data * right->data,
        std::vector{ left, right },
        "*"
    );

    auto const* out_ptr = out.get();
    out->backward = [left, right, out_ptr]() {
        left->grad += right->data * out_ptr->grad;
        right->grad += left->data * out_ptr->grad;
    };

    return out;
}

// Multiplication overload for constant left parameter
auto operator*(double left, const ValuePtr& right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left * right->data,
        std::vector{ right },
        "*" // std::format("* {:.2f}", left)
    );

    auto const* out_ptr = out.get();
    out->backward = [left, right, out_ptr]() {
        right->grad += left * out_ptr->grad;
    };

    return out;
}

// Multiplication overload for constant right parameter
auto operator*(const ValuePtr& left, double right) -> ValuePtr {
    auto out = std::make_shared<Value>(
        left->data * right,
        std::vector{ left },
        "*" // std::format("* {:.2f}", right)
    );

    auto const* out_ptr = out.get();
    out->backward = [left, right, out_ptr]() {
        left->grad += right * out_ptr->grad;
    };

    return out;
}

auto operator/(const ValuePtr& left, const ValuePtr& right) -> ValuePtr {
    return left * pow(right, std::make_shared<Value>(-1.0));
}

// Division overload for constant left parameter
auto operator/(double left, const ValuePtr& right) -> ValuePtr {
    return left * pow(right, std::make_shared<Value>(-1.0));
}

// Division overload for constant right parameter
auto operator/(const ValuePtr& left, double right) -> ValuePtr {
    return left * (1.0 / right);
}

auto operator-(const ValuePtr& v) -> ValuePtr {
    return -1.0 * v;
}


auto pow(const ValuePtr& base, const ValuePtr& exp) -> ValuePtr {
    // x**y
    auto x = base->data;
    auto y = exp->data;

    auto out = std::make_shared<Value>(
        std::pow(x, y),
        std::vector{ base, exp },
        "pow"
    );

    auto const* out_ptr = out.get();
    out->backward = [base, exp, x, y, out_ptr]() {
        base->grad += (y * std::pow(x, y - 1)) * out_ptr->grad;
        exp->grad += (out_ptr->data * std::log(x)) * out_ptr->grad;
    };   
    
    return out;
}

// Pow overload with constant exponent
auto pow(const ValuePtr& base, double exp) -> ValuePtr {
    // x**y
    auto x = base->data;

    auto out = std::make_shared<Value>(
        std::pow(x, exp),
        std::vector{ base },
        "pow" // std::format("pow (exp: {:.2f})", exp)
    );

    auto const* out_ptr = out.get();
    out->backward = [base, exp, x, out_ptr]() {
        base->grad += (exp * std::pow(x, exp - 1)) * out_ptr->grad;
    };   
    
    return out;
}


auto exp(const ValuePtr& v) -> ValuePtr {
    auto x = v->data;

    auto out = std::make_shared<Value>(
        std::exp(x),
        std::vector{ v },
        "exp"
    );

    auto const* out_ptr = out.get();
    out->backward = [v, out_ptr]() {
        v->grad += out_ptr->data * out_ptr->grad;
    };   
    
    return out;
}


auto tanh(const ValuePtr& v) -> ValuePtr {
    auto x = v->data;
    auto t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    
    auto out = std::make_shared<Value>(
        t,
        std::vector{ v },
        "tanh"
    );

    auto const* out_ptr = out.get();
    out->backward = [v, t, out_ptr]() {
        v->grad += (1 - t * t) * out_ptr->grad;
    };    

    return out;
}


auto relu(const ValuePtr& v) -> ValuePtr {
    auto out = std::make_shared<Value>(
        v->data < 0.0 ? 0.0 : v->data,
        std::vector{ v },
        "relu"
    );

    auto const* out_ptr = out.get();
    out->backward = [v, out_ptr]() {
        v->grad += (out_ptr->data > 0 ? 1.0 : 0.0) * out_ptr->grad;
    };

    return out;
}


} // namespace micrograd
