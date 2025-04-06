#ifndef PARTITION_HPP
#define PARTITION_HPP

#include "arch/mesh.hpp"
// spatial * temporal * sharing = block num

struct Mesh {
  // Number of the core
  int coreNum;

  // Footprint for one core
  int footprintPerCore;

  // Bandwidth of onchip memory
  int onchipBandwidth;

  // Bandwidth of offchip memory
  int offchipBandwidth;
};

using PartitionVector =
    std::unordered_map<DNN::Dimension, std::tuple<int, int, int>,
                       DNN::DimensionHash>;

class PartitionAnalysis {
 public:
  PartitionAnalysis(const std::shared_ptr<DNN::OperatorGroup> _group,
                    const std::shared_ptr<Mesh> _mesh)
      : group(_group), mesh(_mesh) {
    std::tie(operators, tensors, dimensions, internalTensors, externalTensors) =
        group->getGroupInfo();
  }

  // Set the partition vector of each dimension
  void setPartitionVector(const PartitionVector _p,
                          const std::vector<DNN::Dimension> _o) noexcept {
    partitionVector = _p;
    orderedDimensions = _o;
  }

  int evaluate() const noexcept {}

  // Get the tile size of each tensor
  int getTileSize(const DNN::Tensor& tensor) const noexcept {
    int tile_size = 1;

    for (auto& dim : tensor.getDimensions()) {
      auto [spatial, temporal, sharing] = partitionVector.at(dim);
      int blockNum = spatial * temporal * sharing;

      tile_size *= dim.getSize() / blockNum;
    }

    return tile_size;
  }

  // Calculate the reduction cost of each operator
  int partitionReductionCost() const noexcept {
    int cost = 0;

    for (const auto& op : operators) {
      auto reduction_dims = op.getReductionDimensions();

      for (const auto& dim : reduction_dims) {
        auto [spatial, temporal, sharing] = partitionVector.at(dim);

        // only consider dimensions that are partitioned spatially
        if (spatial == 1) continue;

        // find reduce tensor
        for (const auto& tensor : op.getOutputs()) {
          auto tensor_dims = tensor.getDimensions();
          if (!std::count(tensor_dims.begin(), tensor_dims.end(), dim))
            continue;

          // all-reduce cost
          int coreGroupNum = mesh->coreNum / spatial;
          int blockSize = getTileSize(tensor);

          int traffic = blockSize * (coreGroupNum - 1);

          // tensor reduce cost
          cost += traffic / mesh->onchipBandwidth;
        }
      }
    }

    return cost;
  }

  // Calculate the traffic of each tensor among opeartor group
  std::pair<int, int> calculatePartitionTraffic() const noexcept {
    int onchip_cost = 0;
    int offchip_cost = 0;

    for (const auto& op : operators) {
      // Calculate the traffic of each tensor
      for (const auto& tensor : op.getTensors()) {
        // Check if the tensor is fused, which means it is consumed by other in
        // lcoal buffer
        bool is_tensor_fused =
            std::count(internalTensors.begin(), internalTensors.end(), tensor);

        if (is_tensor_fused)
          // Skip internal tensors
          continue;

        int tile_size = getTileSize(tensor);

        int onchip_traffic = tile_size;
        int offchip_traffic = tile_size;

        bool access_tensor = false;
        auto tensor_dims = tensor.getDimensions();

        for (auto& dim : orderedDimensions) {
          // From inner loop to outer loop

          bool dim_in_tensor =
              std::count(tensor_dims.begin(), tensor_dims.end(), dim);
          // Now we only consider the outer loop
          if (dim_in_tensor) access_tensor = true;

          // Skip the inner loop
          if (!access_tensor) continue;

          auto [spatial, temporal, sharing] = partitionVector.at(dim);

          // Calculate the traffic of sharing tensor (or sharing == 1)
          onchip_traffic *= temporal * (sharing - 1);
          offchip_traffic *= temporal;
        }

        // Calculate the traffic latency of spatial tensor
        onchip_cost += onchip_traffic / mesh->onchipBandwidth;
        offchip_cost += offchip_traffic / mesh->offchipBandwidth;
      }
    }

    return {onchip_cost, offchip_cost};
  }

  // Calculate the footprint of each the opeartor group
  int calculatePartitionFootprint() const noexcept {
    int volume = 0;

    for (const auto& op : operators) {
      // Collect the dimensions of the operator
      std::unordered_set<DNN::Dimension, DNN::DimensionHash> op_dims;

      for (const auto& tensor : op.getTensors())
        for (const auto& dim : tensor.getDimensions()) op_dims.insert(dim);

      // Calculate the footprint of each tensor
      for (const auto& tensor : op.getTensors()) {
        int tile_size = getTileSize(tensor);

        int footprint = tile_size;

        // Check if the tensor is fused, which means it is consumed by other in
        // lcoal buffer
        bool is_tensor_fused =
            std::count(internalTensors.begin(), internalTensors.end(), tensor);

        if (!is_tensor_fused) {
          // Do not require to stage the dimensions of other tensors
          volume += footprint;
          continue;
        }

        auto outputs = op.getOutputs();
        // Calculate the footprint of each tensor if it is the output tensor
        if (!std::count(outputs.begin(), outputs.end(), tensor)) continue;

        auto tensor_dims = tensor.getDimensions();
        bool expand = false;
        for (auto it = orderedDimensions.rbegin();
             it != orderedDimensions.rend(); it++) {
          // From outer loop to inner loop
          auto dim = *it;

          // Get the partition info
          auto [spatial, temporal, sharing] = partitionVector.at(dim);

          if (!std::count(op_dims.begin(), op_dims.end(), dim))
            // Skip the loop of other operators
            continue;

          if (!std::count(tensor_dims.begin(), tensor_dims.end(), dim))
            // The loop dimensions of other tensors nested within it
            expand = true;
          continue;

          if (expand)
            // Extend the outer loops of the fused tensor
            footprint *= temporal * sharing;

          volume += footprint;
        }
      }
    }

    return volume;
  }

  auto getOperatorGroup() const noexcept { return group; }

 private:
  // Mesh
  std::shared_ptr<Mesh> mesh;

  // Partition Vector
  PartitionVector partitionVector;

  // Ordered dimensions of tile loop
  std::vector<DNN::Dimension> orderedDimensions;

  // Operator Group
  std::shared_ptr<DNN::OperatorGroup> group;

  // Operators
  std::vector<DNN::Operator> operators;

  // Tensors
  std::unordered_set<DNN::Tensor, DNN::TensorHash> tensors;

  // Dimensions
  std::unordered_set<DNN::Dimension, DNN::DimensionHash> dimensions;

  // Internal tensors
  std::unordered_set<DNN::Tensor, DNN::TensorHash> internalTensors;

  // External tensors
  std::unordered_set<DNN::Tensor, DNN::TensorHash> externalTensors;
};

#endif