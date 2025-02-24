#include "dnn/dag.hpp"

int main() {
  // Define the dimensions
  Dimension b("b", 1);
  Dimension h("h", 16);
  Dimension m("m", 64);
  Dimension n("n", 64);
  Dimension k("k", 64);
  Dimension l("l", 64);

  // Define the tensors
  Tensor tQ("tQ", b, h, m, k);
  Tensor tK("tK", b, h, k, n);
  Tensor tA("tA", b, h, m, n);
  Tensor tV("tV", b, h, n, l);
  Tensor tO("tO", b, h, m, l);

  // Define the operator
  Operator mm0("MatMul0", {tQ, tK}, {tA});
  Operator mm1("MatMul1", {tA, tV}, {tO});

  // Define the DAG
  DAG operatorGraph(mm0, mm1);
}