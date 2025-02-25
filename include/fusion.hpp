#ifndef FUSION_HPP
#define FUSION_HPP

#include <algorithm>

#include "dnn/dag.hpp"
#include "dnn/group.hpp"

class FusionSpace {
 public:
  FusionSpace(std::shared_ptr<DNN::DAG> _operatorGraph)
      : operatorGraph(_operatorGraph) {}

  auto generateOperatorGroups(
      std::vector<std::unordered_set<DNN::Operator, DNN::OperatorHash>>
          connected) {
    std::vector<std::shared_ptr<DNN::OperatorGroup>> opGroups;

    for (const auto &con : connected) {
      // con: std::unordered_set<DNN::Operator, DNN::OperatorHash>

      // Initialize the operator group
      auto opGroup = std::make_shared<DNN::OperatorGroup>(operatorGraph);
      for (const auto &op : con) {
        // Find the operator
        opGroup->addOperator(op);
      }
      opGroup->construct();
      opGroups.push_back(opGroup);
    }

    return opGroups;
  }

  void fuseStrategy() {
    // Randomly fuse operators

    // Get the number of tensors
    int tensor_num = operatorGraph->getNumTensors();
    auto fusion_bit = std::vector<bool>(tensor_num, false);

    // Randomly select the tensors to fuse
    srand(time(0));
    for (int i = 0; i < tensor_num; i++) {
      fusion_bit[i] = static_cast<bool>(rand() % 2);
    }

    // Fuse the selected tensors
    operatorGraph->setTensorFusionStatus(fusion_bit);

    // Get the operator groups
    operatorGraph->connectOperators();
    auto connected = operatorGraph->findConnectedComponents();
    auto operators = operatorGraph->getOperators();

    auto groups = generateOperatorGroups(connected);
  }

 private:
  std::shared_ptr<DNN::DAG> operatorGraph;
};
#endif