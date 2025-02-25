#ifndef DNN_DIMENSION_HPP
#define DNN_DIMENSION_HPP

#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace DNN {
class Dimension {
 public:
  Dimension(std::string _name, int _size) : name(_name), size(_size) {}

  // Compare two dimensions
  bool operator<(const Dimension& other) const { return name < other.name; }

  // Compare two dimensions
  bool operator==(const Dimension& other) const { return name == other.name; }

  // Get the name of the dimension
  auto getName() const noexcept { return name; }

 private:
  // Name of the dimension
  std::string name;

  // Size of the dimension
  int size;
};

struct DimensionHash {
  size_t operator()(const Dimension& d) const {
    return std::hash<std::string>()(d.getName());
  }
};
};  // namespace DNN
#endif