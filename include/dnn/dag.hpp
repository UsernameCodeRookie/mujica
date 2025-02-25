#ifndef DNN_DAG_HPP
#define DNN_DAG_HPP

#include <map>
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
        fused[input.getName()] = false;
      }

      for (const auto &output : op.getOutputs()) {
        fused[output.getName()] = false;
      }
    }
  }

  // Get the number of tensors
  auto getNumTensors() const noexcept { return fused.size(); }

  // Get the operators
  auto getOperators() const noexcept { return operators; }

  // Set the fusion status of a tensor
  void setTensorFusionStatus(const std::vector<bool> &fusionBit) noexcept {
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
            if (input == output)
              edges[std::make_pair(op0.getName(), op1.getName())] = input;

            if (input == output and fused[input.getName()] == true)
              fusionEdges[std::make_pair(op0.getName(), op1.getName())] = input;
          }
  }

  // Find the connected components
  auto findConnectedComponents() {
    std::vector<std::unordered_set<std::string>> connectedComponents;
    std::unordered_set<std::string> visited;

    for (const auto &op : operators) {
      const std::string &opName = op.getName();
      if (visited.find(opName) == visited.end()) {
        std::unordered_set<std::string> component;
        dfs(opName, visited, component);
        connectedComponents.push_back(component);
      }
    }
    return connectedComponents;
  }

  // Depth-first search on the fusion edges
  void dfs(const std::string &opName, std::unordered_set<std::string> &visited,
           std::unordered_set<std::string> &component) {
    visited.insert(opName);
    component.insert(opName);

    for (const auto &op : operators) {
      if (fusionEdges.find({opName, op.getName()}) != fusionEdges.end() ||
          fusionEdges.find({op.getName(), opName}) != fusionEdges.end()) {
        if (visited.find(op.getName()) == visited.end()) {
          dfs(op.getName(), visited, component);
        }
      }
    }
  }

 private:
  // Operators in the DAG
  std::vector<Operator> operators;

  // Tensors fusion map
  std::map<std::string, bool> fused;

  // Edges in the DAG
  std::map<std::pair<std::string, std::string>, Tensor> edges;

  // Fusion edges in the DAG
  std::map<std::pair<std::string, std::string>, Tensor> fusionEdges;
};
};  // namespace DNN
#endif