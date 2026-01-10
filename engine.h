#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <format>


namespace micrograd {


class Value;
using ValuePtr = std::shared_ptr<Value>;


class Value {
public:
    double data;
    double grad;
    std::vector<ValuePtr> children;
    std::string op;
    std::function<void()> backward;

public:
    Value(
        double data,
        std::vector<ValuePtr> children = {},
        std::string op = ""
    ):
        data{ data },
        grad{ 0.0 },
        children{ std::move(children) },
        op{ std::move(op) },
        backward{ []() {} }
    {}

    auto print_graph(std::size_t depth = 0) const -> void;

    friend auto operator<<(std::ostream& stream, const Value& value) -> std::ostream& {
        stream << std::format("Value({}), op='{}'", value.data, value.op); 
        return stream;
    }
};

// Compute gradient for computational graph starting with root node
// based on topological sort
auto backward(const ValuePtr& root) -> void;


[[nodiscard]] auto operator+(const ValuePtr& left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator+(double left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator+(const ValuePtr& left, double right) -> ValuePtr;

[[nodiscard]] auto operator-(const ValuePtr& left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator-(double left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator-(const ValuePtr& left, double right) -> ValuePtr;

[[nodiscard]] auto operator*(const ValuePtr& left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator*(double left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator*(const ValuePtr& left, double right) -> ValuePtr;

[[nodiscard]] auto operator/(const ValuePtr& left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator/(double left, const ValuePtr& right) -> ValuePtr;
[[nodiscard]] auto operator/(const ValuePtr& left, double right) -> ValuePtr;

[[nodiscard]] auto operator-(const ValuePtr& v) -> ValuePtr;

[[nodiscard]] auto pow(const ValuePtr& base, const ValuePtr& exp) -> ValuePtr;
[[nodiscard]] auto pow(const ValuePtr& base, double exp) -> ValuePtr;
[[nodiscard]] auto exp(const ValuePtr& v) -> ValuePtr;

[[nodiscard]] auto tanh(const ValuePtr& v) -> ValuePtr;
[[nodiscard]] auto relu(const ValuePtr& v) -> ValuePtr;


} // namespace micrograd
