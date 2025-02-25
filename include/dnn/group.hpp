#ifndef DNN_GROUP_HPP
#define DNN_GROUP_HPP

#include "operator.hpp"

namespace DNN {
class OperatorGroup {
 public:
  OperatorGroup() = default;

  void AddOperator(const Operator &op) noexcept { operators.push_back(op); }

 private:
  std::vector<Operator> operators;
};

};  // namespace DNN
#endif