#ifndef DNN_GROUP_HPP
#define DNN_GROUP_HPP

#include "dag.hpp"

namespace DNN {
class OperatorGroup {
 public:
  OperatorGroup(std::shared_ptr<DAG> _graph) : graph(_graph) {}

  void AddOperator(const Operator &op) noexcept { operators.push_back(op); }

  void classifyTensorsByTopology() noexcept {
    auto tensors = std::unordered_set<Tensor, TensorHash>();
  }

 private:
  // Operators in the group
  std::vector<Operator> operators;

  // DAG
  std::shared_ptr<DAG> graph;
};

};  // namespace DNN
#endif