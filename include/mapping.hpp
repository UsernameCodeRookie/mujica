#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <functional>
#include <limits>

#include "algo/genetic.hpp"
#include "partition.hpp"

class PartitionIndividual : public Algorithm::IIndividual {
 public:
  PartitionIndividual(
      const std::vector<DNN::Dimension> _dims,
      const std::function<int(PartitionVector, std::vector<DNN::Dimension>)>
          _eval,
      const std::function<int(PartitionVector, std::vector<DNN::Dimension>)>
          _cons)
      : dims(_dims), evaluate(_eval), constraint(_cons) {
    randomize();
  }

  void randomize() {
    // Initialize with random partition and order
    p.clear();
    o.clear();

    for (auto dim : dims) {
      p[dim] = std::make_tuple(rand() % 4 + 1, rand() % 4 + 1, rand() % 4 + 1);
      o.push_back(dim);
    }
    std::random_shuffle(o.begin(), o.end());
  }

  int fitness() const override {
    if (constraint(p, o)) return std::numeric_limits<int>::max();

    return evaluate(p, o);
  }

  void mutate() override {
    if (p.empty()) return;
    auto it = p.begin();
    std::advance(it, rand() % p.size());
    int a = rand() % 4 + 1, b = rand() % 4 + 1, c = rand() % 4 + 1;
    it->second = std::make_tuple(a, b, c);
    std::random_shuffle(o.begin(), o.end());
  }

  std::shared_ptr<IIndividual> clone() const override {
    auto new_obj = std::make_shared<PartitionIndividual>(*this);
    return new_obj;
  }

  std::shared_ptr<IIndividual> crossover(
      const std::shared_ptr<IIndividual>& other) const override {
    auto other_part = std::dynamic_pointer_cast<PartitionIndividual>(other);
    auto child = std::make_shared<PartitionIndividual>();

    // Uniform crossover
    for (const auto& dim : dims) {
      if (rand() % 2) {
        child->p[dim] = this->p.at(dim);
      } else {
        child->p[dim] = other_part->p.at(dim);
      }
    }

    child->o = this->o;
    if (rand() % 2) std::random_shuffle(child->o.begin(), child->o.end());

    return child;
  }

  void print() const override {
    std::cout << "Fitness: " << fitness() << "\n";
    for (auto dim : o) {
      auto [x, y, z] = p.at(dim);
      //   std::cout << DNN::to_string(dim) << ": (" << x << ", " << y << ", "
      //   << z
      //             << ")\n";
    }
  }

 private:
  // Dimensions
  std::vector<DNN::Dimension> dims;

  // Evaluation function
  std::function<int(PartitionVector, std::vector<DNN::Dimension>)> evaluate;

  // Constraint function
  std::function<int(PartitionVector, std::vector<DNN::Dimension>)> constraint;

  // Partition vector
  PartitionVector p;

  // Ordered dimensions
  std::vector<DNN::Dimension> o;
};

#endif