#ifndef DNN_GROUP_HPP
#define DNN_GROUP_HPP

#include "dag.hpp"

namespace DNN {
class OperatorGroup {
 public:
  OperatorGroup(std::shared_ptr<DAG> _graph) : graph(_graph) {}

  void addOperator(const Operator &op) noexcept { operators.push_back(op); }

  void collectTensors() noexcept {
    // Get the tensors
    for (const auto &op : operators) {
      for (const auto &t : op.getInputs()) {
        tensors.insert(t);
      }

      for (const auto &t : op.getOutputs()) {
        tensors.insert(t);
      }
    }
  }

  void collectDimensions() noexcept {
    // Get the dimensions
    for (const auto &t : tensors) {
      for (const auto &d : t.getDimensions()) {
        dimensions.insert(d);
      }
    }
  }

  void classifyTensorsByTopology() noexcept {
    // Classify the tensors
    for (const auto &t : tensors) {
      auto edge = graph->FindOperatorPair(t);
      if (!edge) continue;

      auto [src_op, dst_op] = *edge;

      bool src_op_in_group =
          std::count(operators.begin(), operators.end(), src_op);
      bool dst_op_in_group =
          std::count(operators.begin(), operators.end(), dst_op);

      if (src_op_in_group and dst_op_in_group) {
        internalTensors.insert(t);
      } else {
        externalTensors.insert(t);
      }
    }
  }

  void construct() noexcept {
    // Clear the tensors and dimensions set
    tensors.clear();
    dimensions.clear();
    internalTensors.clear();
    externalTensors.clear();

    // Collect the tensors and dimensions
    collectTensors();
    collectDimensions();
    classifyTensorsByTopology();
  }

  auto getGroupInfo() const noexcept {
    return std::make_tuple(operators, tensors, dimensions, internalTensors,
                           externalTensors);
  }

 private:
  // Operators in the group
  std::vector<Operator> operators;

  // Tensors in the group
  std::unordered_set<Tensor, TensorHash> tensors;

  // Internal tensors
  std::unordered_set<Tensor, TensorHash> internalTensors;

  // External tensors
  std::unordered_set<Tensor, TensorHash> externalTensors;

  // Dimensions in the group
  std::unordered_set<Dimension, DimensionHash> dimensions;

  // DAG
  std::shared_ptr<DAG> graph;
};

};  // namespace DNN
#endif