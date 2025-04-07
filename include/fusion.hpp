#ifndef FUSION_HPP
#define FUSION_HPP

#include <algorithm>
#include <functional>
#include <limits>

#include "dnn/dag.hpp"
#include "dnn/group.hpp"
#include "mapper.hpp"

class RandomSearch {
 public:
  RandomSearch(int _numIterations) : numIterations(_numIterations) {
    srand(time(0));
  }

  auto search(const std::vector<bool> &design_space,
              const std::function<int(std::vector<bool>)> eval) const noexcept {
    // Randomly search the fusion space
    int best_score = std::numeric_limits<int>::infinity();
    std::vector<bool> solution;

    for (int i = 0; i < numIterations; i++) {
      // Randomly select the tensors to fuse
      auto candidate = std::vector<bool>(design_space.size(), false);
      for (int j = 0; j < static_cast<int>(design_space.size()); j++) {
        candidate[j] = static_cast<bool>(rand() % 2);
      }

      // Evaluate the fusion strategy
      auto score = eval(candidate);

      // Update the best solution
      if (score > best_score) continue;

      best_score = score;
      solution = candidate;
    }

    return solution;
  }

 private:
  int numIterations;
};

class TraverseSearch {
 public:
  TraverseSearch() = default;

  auto search(const std::vector<bool> &design_space,
              const std::function<int(std::vector<bool>)> eval) const noexcept {
    int best_score = std::numeric_limits<int>::infinity();
    std::vector<bool> solution;

    int size = static_cast<int>(design_space.size());
    int combinations = 1 << size;

    for (int i = 0; i < combinations; i++) {
      // Traverse the combinations
      auto candidate = std::vector<bool>(size, false);

      for (int j = 0; j < size; j++) {
        candidate[j] = (i >> j) & 1;
      }

      // Evaluate the fusion strategy
      auto score = eval(candidate);

      // Update the best solution
      if (score > best_score) continue;

      best_score = score;
      solution = candidate;
    }

    return solution;
  }
};

class FusionSpace {
 public:
  FusionSpace(const std::shared_ptr<DNN::DAG> _operatorGraph)
      : operatorGraph(_operatorGraph) {}

  auto generateOperatorGroups(
      const std::vector<std::unordered_set<DNN::Operator, DNN::OperatorHash>>
          &connected) const noexcept {
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

  // TODO: implement the searchFusionSpace function
  void searchFusionSpace(
      const std::shared_ptr<Architecture::Mesh> mesh) const noexcept {
    // Randomly fuse operators

    // Get the number of tensors
    operatorGraph->connectOperators();
    int tensor_num = operatorGraph->getNumPotentialFusionTensors();
    auto fusion_bit = std::vector<bool>(tensor_num, false);

    // Traverse the search space
    TraverseSearch ts;

    auto eval = [&](const std::vector<bool> &fusion_bit) -> int {
      // Evaluate the fusion strategy

      // Fuse the selected tensors
      operatorGraph->setTensorFusionStatus(fusion_bit);

      // Get the operator groups
      operatorGraph->connectFusionOperators();
      auto connected = operatorGraph->findConnectedComponents();
      auto operators = operatorGraph->getOperators();

      auto groups = generateOperatorGroups(connected);

      auto analysis = std::make_shared<PartitionAnalysis>(groups, mesh);
      auto mapper = std::make_shared<Mapper>(analysis);

      mapper->search();

      return 0;
    };

    auto best_solution = ts.search(fusion_bit, eval);
  }

 private:
  std::shared_ptr<DNN::DAG> operatorGraph;
};
#endif