#include "fusion.hpp"

int main() {
  // Define the dimensions
  DNN::Dimension b("b", 1);
  DNN::Dimension h("h", 16);
  DNN::Dimension m("m", 64);
  DNN::Dimension n("n", 64);
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
  DNN::DAG operatorGraph(mm0, mm1);

  // Fusion space
  FusionSpace fs(operatorGraph);

  fs.FuseStrategy();
}