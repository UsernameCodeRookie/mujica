#ifndef FUSION_HPP
#define FUSION_HPP

#include <algorithm>

#include "dnn/dag.hpp"
#include "dnn/group.hpp"

class FusionSpace {
 public:
  FusionSpace(DNN::DAG _operatorGraph) : operatorGraph(_operatorGraph) {}

  auto GenerateOperatorGroups(
      std::vector<DNN::Operator> operators,
      std::vector<std::unordered_set<std::string>> connected) {
    std::vector<DNN::OperatorGroup> opGroups;

    for (const auto &con : connected) {
      // con: std::unordered_set<std::string>

      // Initialize the operator group
      DNN::OperatorGroup opGroup;
      for (const auto &op_name : con) {
        // Find the operator
        auto it = std::find_if(
            operators.begin(), operators.end(),
            [&](const DNN::Operator &op) { return op.getName() == op_name; });

        if (it == operators.end()) continue;

        // Add the operator to the group
        opGroup.AddOperator(*it);
      }
      opGroups.push_back(opGroup);
    }

    return opGroups;
  }

  void FuseStrategy() {
    // Randomly fuse operators

    // Get the number of tensors
    int tensor_num = operatorGraph.getNumTensors();
    auto fusion_bit = std::vector<bool>(tensor_num, false);

    // Randomly select the tensors to fuse
    for (int i = 0; i < tensor_num; i++) {
      fusion_bit[i] = rand() % 2;
    }

    // Fuse the selected tensors
    operatorGraph.setTensorFusionStatus(fusion_bit);

    // Get the operator groups
    operatorGraph.connectOperators();
    auto connected = operatorGraph.findConnectedComponents();
    auto operators = operatorGraph.getOperators();

    auto groups = GenerateOperatorGroups(operators, connected);
  }

 private:
  DNN::DAG operatorGraph;
};
#endif