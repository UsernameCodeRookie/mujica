#ifndef DNN_TENSOR_HPP
#define DNN_TENSOR_HPP

#include <string>
#include <vector>

#include "dimension.hpp"

namespace DNN {
class Tensor {
 public:
  template <typename... Dims>
  Tensor(std::string _name, Dims... dims) : name(_name), dimensions{dims...} {}

  Tensor() : name("null"), dimensions{} {}

  // Get the name
  auto getName() const noexcept { return name; }

  // Get the dimensions
  auto getDimensions() const noexcept { return dimensions; }

  // Compare two tensors
  bool operator==(const Tensor& other) const { return name == other.name; }

  bool operator<(const Tensor& other) const { return name < other.name; }

 private:
  // Name of the tensor
  std::string name;

  // Dimensions of the tensor
  std::vector<Dimension> dimensions;
};

struct TensorHash {
  size_t operator()(const Tensor& t) const {
    return std::hash<std::string>()(t.getName());
  }
};

};  // namespace DNN
#endif