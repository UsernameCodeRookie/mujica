#include <set>

#include "tensor.hpp"

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
    std::set<Dimension> reductDims;

    for (const auto &t : inputs) {
      for (const auto &d : t.getDimensions()) {
        reductDims.insert(d);
      }
    }

    for (const auto &t : outputs) {
      for (const auto &d : t.getDimensions()) {
        if (reductDims.find(d) != reductDims.end()) {
          reductDims.erase(d);
        }
      }
    }

    return reductDims;
  }

  // Get the inputs
  auto getInputs() const noexcept { return inputs; }

  // Get the outputs
  auto getOutputs() const noexcept { return outputs; }

  // Get the name
  auto getName() const noexcept { return name; }

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