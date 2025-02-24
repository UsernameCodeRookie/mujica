#include <iostream>
#include <map>

#include "operator.hpp"

class DAG {
 public:
  template <typename... Ops>
  DAG(Ops... _ops) : operators{_ops...} {
    // Collect the tensors
    for (const auto &op : operators) {
      for (const auto &input : op.getInputs()) {
        tensors.insert(input);
      }

      for (const auto &output : op.getOutputs()) {
        tensors.insert(output);
      }
    }
  }

  // Get the tensors
  auto getTensors() const noexcept { return tensors; }

  // Connect the operators
  void connectFusionOperators() noexcept {
    for (const auto &op0 : operators)
      for (const auto &input : op0.getInputs())
        for (const auto &op1 : operators)
          for (const auto &output : op1.getOutputs())
            if (input == output)
              edges[std::make_pair(op0.getName(), op1.getName())] = input;
  }

 private:
  // Operators in the DAG
  std::vector<Operator> operators;

  // Tensors in the DAG
  std::set<Tensor> tensors;

  // Edges in the DAG
  std::map<std::pair<std::string, std::string>, Tensor> edges;
};