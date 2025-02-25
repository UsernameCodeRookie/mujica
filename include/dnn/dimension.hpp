#ifndef DNN_DIMENSION_HPP
#define DNN_DIMENSION_HPP

#include <iostream>
#include <memory>
#include <string>

namespace DNN {
class Dimension {
 public:
  Dimension(std::string _name, int _size) : name(_name), size(_size) {}

  // Compare two dimensions
  bool operator<(const Dimension& other) const { return name < other.name; }

 private:
  // Name of the dimension
  std::string name;

  // Size of the dimension
  int size;
};
};  // namespace DNN
#endif