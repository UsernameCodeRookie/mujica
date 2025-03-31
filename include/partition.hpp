#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "arch/mesh.hpp"

struct Partition {
  int spatial;
  int temporal;
  int sharing;
};
// spatial * temporal * sharing = block num

class PartitionAnalysis {
 public:
 private:
  // Partition Vector
  std::unordered_map<DNN::Dimension, Partition, DNN::DimensionHash>
      partitionVector;
};

#endif