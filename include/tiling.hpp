#ifndef TILLING_HPP
#define TILLING_HPP

#include "dnn/group.hpp"

class TilingAnalysis {
 public:
  TilingAnalysis(std::shared_ptr<DNN::OperatorGroup> _group) : group(_group) {}

  void generateTilingSpace() {
    auto [operators, tensors, dimensions, internalTensors, externalTensors] =
        group->getGroupInfo();
    // TODO: implement by SCIP to split tensor dimension to tile
    // TODO: cost model to evaluate the tiling space
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