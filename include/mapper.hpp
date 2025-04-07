#ifndef MAPPER_HPP
#define MAPPER_HPP

#include "mapping.hpp"

class Mapper {
 public:
  Mapper(const std::shared_ptr<PartitionAnalysis> _analysis)
      : analysis(_analysis) {}

  void search() const noexcept {
    auto group = analysis->getOperatorGroup();
    auto [operators, tensors, dimensions, internalTensors, externalTensors] =
        group->getGroupInfo();

    auto dims =
        std::vector<DNN::Dimension>(dimensions.begin(), dimensions.end());

    auto eval = [&](PartitionVector p, std::vector<DNN::Dimension> o) -> int {
      analysis->setPartitionVector(p, o);
      return analysis->evaluate();
    };

    auto cons = [&](PartitionVector p, std::vector<DNN::Dimension> o) -> bool {
      analysis->setPartitionVector(p, o);
      return analysis->constraint();
    };

    auto ga = std::make_shared<Algorithm::GeneticAlgorithm>(30, 50, 0.3f, 0.7f);

    ga->initialize<PartitionIndividual>(dims, eval, cons);
    ga->run();
  }

 private:
  std::shared_ptr<PartitionAnalysis> analysis;
};

#endif