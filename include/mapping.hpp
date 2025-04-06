#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <functional>

#include "algo/genetic.hpp"
#include "partition.hpp"

class PartitionIndividual : public Algorithm::IIndividual {
 public:
  PartitionIndividual(
      const std::vector<DNN::Dimension> _dims,
      const std::function<int(PartitionVector, std::vector<DNN::Dimension>)>
          _eval)
      : dims(_dims), evaluate(_eval) {
    randomize();
  }

  void randomize() {
    // Initialize with random partition and order
    partitionVector.clear();
    orderedDimensions.clear();

    for (auto dim : dims) {
      partitionVector[dim] =
          std::make_tuple(rand() % 4 + 1, rand() % 4 + 1, rand() % 4 + 1);
      orderedDimensions.push_back(dim);
    }
    std::random_shuffle(orderedDimensions.begin(), orderedDimensions.end());
  }

  int fitness() const override {
    return evaluate(partitionVector, orderedDimensions);
  }

  void mutate() override {
    if (partitionVector.empty()) return;
    auto it = partitionVector.begin();
    std::advance(it, rand() % partitionVector.size());
    int a = rand() % 4 + 1, b = rand() % 4 + 1, c = rand() % 4 + 1;
    it->second = std::make_tuple(a, b, c);
    std::random_shuffle(orderedDimensions.begin(), orderedDimensions.end());
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
        child->partitionVector[dim] = this->partitionVector.at(dim);
      } else {
        child->partitionVector[dim] = other_part->partitionVector.at(dim);
      }
    }

    child->orderedDimensions = this->orderedDimensions;
    if (rand() % 2)
      std::random_shuffle(child->orderedDimensions.begin(),
                          child->orderedDimensions.end());

    return child;
  }

  void print() const override {
    std::cout << "Fitness: " << fitness() << "\n";
    for (auto dim : orderedDimensions) {
      auto [x, y, z] = partitionVector.at(dim);
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

  // Partition vector
  PartitionVector partitionVector;

  // Ordered dimensions
  std::vector<DNN::Dimension> orderedDimensions;
};

#endif