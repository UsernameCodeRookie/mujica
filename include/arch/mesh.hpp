#ifndef MESH_HPP
#define MESH_HPP

#include <vector>

namespace Architecture {
enum MeshType {
  Mesh2D,
  Mesh3D,
  Ring,
  Tree,
  Hypercube,
  Torus,
};

class Mesh {
 public:
  Mesh(const std::vector<int>& _shape, int _bandwidth)
      : shape(_shape), bandwidth(_bandwidth) {}

  virtual int allGatherCost() const noexcept = 0;

  virtual int allReduceCost() const noexcept = 0;

 private:
  // mesh shape
  std::vector<int> shape;

  // bandwidth
  int bandwidth;
};
}  // namespace Architecture

#endif