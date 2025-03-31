#include "fusion.hpp"
#include "partition.hpp"

int main() {
  // TODO: design a better input format

  // Define the dimensions
  DNN::Dimension b("b", 1);
  DNN::Dimension h("h", 12);
  DNN::Dimension m("m", 1024);
  DNN::Dimension n("n", 1024);
  DNN::Dimension k("k", 64);
  DNN::Dimension l("l", 64);

  // Define the tensors
  DNN::Tensor tQ("tQ", b, h, m, k);
  DNN::Tensor tK("tK", b, h, k, n);
  DNN::Tensor tA("tA", b, h, m, n);
  DNN::Tensor tV("tV", b, h, n, l);
  DNN::Tensor tO("tO", b, h, m, l);

  // Define the operator
  DNN::Operator mm0("MatMul0", {tQ, tK}, {tA});
  DNN::Operator mm1("MatMul1", {tA, tV}, {tO});

  // Define the DAG
  auto operatorGraph = std::make_shared<DNN::DAG>(mm0, mm1);

  // Fusion space
  auto fs = FusionSpace(operatorGraph);

  fs.searchFusionSpace();

  auto mesh = Architecture::Mesh({2, 2}, 32);
}