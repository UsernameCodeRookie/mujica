#ifndef DNN_OPERATOR_HPP
#define DNN_OPERATOR_HPP

#include <set>

#include "tensor.hpp"

namespace DNN {
class Operator {
 public:
  template <typename... InTensors, typename... OutTensors>
  Operator(std::string _name, std::vector<Tensor> _inputs,
           std::vector<Tensor> _outputs)
      : name(_name), inputs(_inputs), outputs(_outputs) {
    reductDims = getReductionDimensions();
  }

  // Get the reduction dimensions
  auto getReductionDimensions() const noexcept {
    std::set<Dimension> reduct_dimensions;

    for (const auto &t : inputs) {
      for (const auto &d : t.getDimensions()) {
        reduct_dimensions.insert(d);
      }
    }

    for (const auto &t : outputs) {
      for (const auto &d : t.getDimensions()) {
        if (reduct_dimensions.find(d) != reduct_dimensions.end()) {
          reduct_dimensions.erase(d);
        }
      }
    }

    return reduct_dimensions;
  }

  // Get the inputs
  auto getInputs() const noexcept { return inputs; }

  // Get the outputs
  auto getOutputs() const noexcept { return outputs; }

  // Get all the tensors
  auto getTensors() const noexcept {
    std::vector<Tensor> tensors;
    tensors.insert(tensors.end(), inputs.begin(), inputs.end());
    tensors.insert(tensors.end(), outputs.begin(), outputs.end());
    return tensors;
  }

  // Get the name
  auto getName() const noexcept { return name; }

  bool operator==(const Operator &other) const { return name == other.name; }

 private:
  // Name of the operator
  std::string name;

  // Input tensors
  std::vector<Tensor> inputs;

  // Output tensors
  std::vector<Tensor> outputs;

  // Reduction dimensions
  std::set<Dimension> reductDims;
};

struct OperatorHash {
  size_t operator()(const Operator &op) const {
    return std::hash<std::string>()(op.getName());
  }
};

struct OperatorPairHash {
  size_t operator()(const std::pair<Operator, Operator> &op) const {
    return std::hash<std::string>()(op.first.getName() + op.second.getName());
  }
};

};  // namespace DNN
#endif