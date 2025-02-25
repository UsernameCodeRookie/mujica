#ifndef TILLING_ANALYSIS_HPP
#define TILLING_ANALYSIS_HPP

#include "dnn/group.hpp"

class TilingAnalysis {
 public:
  TilingAnalysis(std::shared_ptr<DNN::OperatorGroup> _group) : group(_group) {}

  void generateTilingSpace() {
    auto [operators, tensors, dimensions, internalTensors, externalTensors] =
        group->getGroupInfo();
  }

 private:
  struct Tiling {
    int factor;
    int size;
  };

  std::shared_ptr<DNN::OperatorGroup> group;

  std::unordered_map<DNN::Dimension, Tiling, DNN::DimensionHash> tilingSpace;
};

#endif