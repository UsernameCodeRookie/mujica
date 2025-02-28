#ifndef TILLING_HPP
#define TILLING_HPP

#include "dnn/group.hpp"

class TilingAnalysis {
 public:
  TilingAnalysis() {}

  void generateTilingSpace(
      const std::shared_ptr<DNN::OperatorGroup> _group) noexcept {
    group = _group;
    std::tie(operators, tensors, dimensions, internalTensors, externalTensors) =
        group->getGroupInfo();
    // TODO: implement by SCIP to split tensor dimension to tile
    // TODO: cost model to evaluate the tiling space
  }

  // Calculate the traffic of the external tensor
  int calculate_external_tensor_traffic(
      const DNN::Tensor& tensor) const noexcept {
    int traffic = 1;

    // Calculate the size of tensor tile
    for (auto& dim : tensor.getDimensions()) {
      traffic *= dim.getSize();
    }

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

      // Multiply the loop iteration size
      traffic *= dim.getSize() / blockSize.at(dim);
    }

    return traffic;
  }

  // Calculate the traffic of the operator group
  int calculate_traffic() const noexcept {
    int traffic = 0;
    for (auto& op : operators) {
      // Calculate the traffic of each tensor
      for (auto& tensor : op.getTensors()) {
        // Check if the tensor is fused, which means it is consumed by other in
        // lcoal buffer
        bool is_tensor_fused =
            std::count(internalTensors.begin(), internalTensors.end(), tensor);

        if (is_tensor_fused)
          // Skip internal tensors
          continue;

        // Calculate the traffic of each tensor if it is external
        traffic += calculate_external_tensor_traffic(tensor);
      }
    }
    return traffic;
  }

  // Calculate the footprint of the internal tensor
  int calculate_internal_tensor_footprint(
      const DNN::Tensor& tensor,
      const std::unordered_set<DNN::Dimension, DNN::DimensionHash> op_dims,
      int tensor_tile_size) const noexcept {
    // If this tensor is fused, it means that it is entirely temporarily stored
    // in a local buffer and consumed by operators both before and after.
    // Therefore, for the loop dimensions of other tensors nested within it, the
    // outer loops of the fused tensor need to be extended at the loop level
    // corresponding to the nested dimensions. See {DAC ’24: Enabling Multiple
    // Tensor-wise Operator Fusion for Transformer Models on Spatial
    // Accelerators.}

    int footprint = tensor_tile_size;

    auto tensor_dims = tensor.getDimensions();

    bool expand = false;
    for (auto it = orderedDimensions.rbegin(); it != orderedDimensions.rend();
         it++) {
      // From outer loop to inner loop
      auto dim = *it;

      if (!std::count(op_dims.begin(), op_dims.end(), dim))
        // Skip the loop of other operators
        continue;

      if (!std::count(tensor_dims.begin(), tensor_dims.end(), dim))
        // The loop dimensions of other tensors nested within it
        expand = true;
      continue;

      if (expand)
        // Extend the outer loops of the fused tensor
        footprint *= tensor_tile_size / blockSize.at(dim);
    }
  }

  // Calculate the footprint of the operator group
  int calculate_footprint() const noexcept {
    int footprint = 0;
    for (auto& op : operators) {
      // Collect the dimensions of the operator
      std::unordered_set<DNN::Dimension, DNN::DimensionHash> op_dims;

      for (auto& tensor : op.getTensors())
        for (auto& dim : tensor.getDimensions()) op_dims.insert(dim);

      // Calculate the footprint of each tensor
      for (auto& tensor : op.getTensors()) {
        // Calculate the size of tensor tile
        int tensor_tile_size = 1;

        for (auto& dim : tensor.getDimensions()) {
          tensor_tile_size *= dim.getSize();
        }

        // Check if the tensor is fused, which means it is consumed by other in
        // lcoal buffer
        bool is_tensor_fused =
            std::count(internalTensors.begin(), internalTensors.end(), tensor);

        if (!is_tensor_fused) {
          // Do not require to stage the dimensions of other tensors
          footprint += tensor_tile_size;
          continue;
        }

        auto outputs = op.getOutputs();
        if (std::count(outputs.begin(), outputs.end(), tensor)) {
          // Calculate the footprint of each tensor if it is the output tensor
          footprint += calculate_internal_tensor_footprint(tensor, op_dims,
                                                           tensor_tile_size);
        }
      }
    }
  }

 private:
  // Tile loop block size
  std::unordered_map<DNN::Dimension, int, DNN::DimensionHash> blockSize;

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