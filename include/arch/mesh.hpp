#ifndef MESH_HPP
#define MESH_HPP

#include <vector>

namespace Architecture {
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
}  // namespace Architecture

#endif