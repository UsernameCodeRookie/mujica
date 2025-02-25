#ifndef DNN_DAG_HPP
#define DNN_DAG_HPP

#include <unordered_map>
#include <unordered_set>

#include "operator.hpp"

namespace DNN {
class DAG {
 public:
  template <typename... Ops>
  DAG(Ops... _ops) : operators{_ops...} {
    // Collect the tensors
    for (const auto &op : operators) {
      for (const auto &input : op.getInputs()) {
        fused[input] = false;
      }

      for (const auto &output : op.getOutputs()) {
        fused[output] = false;
      }
    }
  }

  // Get the number of tensors
  auto getNumTensors() const noexcept { return fused.size(); }

  // Get the operators
  auto getOperators() const noexcept { return operators; }

  // Clear the fusion status
  void clearFusionStatus() noexcept {
    for (auto &f : fused) {
      f.second = false;
    }
    fusionEdges.clear();
  }

  // Set the fusion status of a tensor
  void setTensorFusionStatus(const std::vector<bool> &fusionBit) noexcept {
    clearFusionStatus();

    int i = 0;
    for (auto &f : fused) {
      f.second = fusionBit[i];
      i++;
    }
  }

  // Connect the operators
  void connectOperators() noexcept {
    for (const auto &op0 : operators)
      for (const auto &input : op0.getInputs())
        for (const auto &op1 : operators)
          for (const auto &output : op1.getOutputs()) {
            if (input == output) edges[std::make_pair(op0, op1)] = input;

            if (input == output and fused[input] == true)
              fusionEdges[std::make_pair(op0, op1)] = input;
          }
  }

  // Find the connected components
  auto findConnectedComponents() {
    std::vector<std::unordered_set<Operator, OperatorHash>> connectedComponents;
    std::unordered_set<Operator, OperatorHash> visited;

    for (const auto &op : operators) {
      if (visited.find(op) == visited.end()) {
        std::unordered_set<Operator, OperatorHash> component;
        dfs(op, visited, component);
        connectedComponents.push_back(component);
      }
    }
    return connectedComponents;
  }

  // Depth-first search on the fusion edges
  void dfs(const Operator &op,
           std::unordered_set<Operator, OperatorHash> &visited,
           std::unordered_set<Operator, OperatorHash> &component) {
    visited.insert(op);
    component.insert(op);

    for (const auto &nop : operators) {
      if (fusionEdges.find({op, nop}) != fusionEdges.end() ||
          fusionEdges.find({nop, op}) != fusionEdges.end()) {
        if (visited.find(nop) == visited.end()) {
          dfs(nop, visited, component);
        }
      }
    }
  }

  // Find the responding operator pair for a tensor
  std::optional<const std::pair<Operator, Operator>> FindOperatorPair(
      const Tensor &t) const noexcept {
    for (const auto &pair : edges) {
      if (pair.second == t) return pair.first;
    }
    return std::nullopt;
  }

 private:
  // Operators in the DAG
  std::vector<Operator> operators;

  // Tensors fusion map
  std::unordered_map<Tensor, bool, TensorHash> fused;

  // Edges in the DAG
  std::unordered_map<std::pair<Operator, Operator>, Tensor, OperatorPairHash>
      edges;

  // Fusion edges in the DAG
  std::unordered_map<std::pair<Operator, Operator>, Tensor, OperatorPairHash>
      fusionEdges;
};

};  // namespace DNN
#endif