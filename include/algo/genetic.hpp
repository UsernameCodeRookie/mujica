#ifndef GENETIC_HPP
#define GENETIC_HPP

#include <algorithm>

#include "dnn/group.hpp"

/*
TODO: Add constraint handling mechanisms to the GeneticAlgorithm framework.
Several typical approaches include:

────────────────────────────────────────────────────────
1. Implicit Constraint Handling via Encoding
   - Design the chromosome representation such that all individuals are
guaranteed to be feasible;
   - Common in continuous domain problems with bounded variables (e.g., x ∈
[lower_bound, upper_bound]);
   - The encoding-to-solution mapping function must always return feasible
values;
   - After mutation or crossover, decoded results must remain valid.

────────────────────────────────────────────────────────
2. Constraint-Preserving Crossover/Mutation Operators
   - Design custom operators that inherently maintain feasibility after
application;
   - Common in permutation-based problems (e.g., TSP), using operators like
Order Crossover (OX), Partially Mapped Crossover (PMX), etc.;
   - Mutation can be designed as swapping two indices to preserve valid
permutations.

────────────────────────────────────────────────────────
3. Penalty Function Method
   - Do not restrict the generation of infeasible individuals, but penalize them
in the fitness function;
   - Ideal for problems with complex or multi-condition constraints;
   - Example: fitness = objective - λ × constraint_violation;
   - λ (penalty coefficient) must be tuned carefully:
     - Too large → algorithm may prematurely converge to feasible local optima;
     - Too small → algorithm may explore infeasible regions too much.

────────────────────────────────────────────────────────
4. Repair After Crossover/Mutation
   - After applying genetic operators, check if the individual is feasible;
   - If not, apply repair strategies (e.g., projection back, clipping to bounds,
or reinitialization);
   - Useful for moderately sized problems.

────────────────────────────────────────────────────────

Suggested Implementation Steps:
   - [ ] Support bounds or constraint information in individual encoding.
   - [ ] Add constraint-checking logic inside mutation/crossover functions.
   - [ ] Optionally implement either repair-based or penalty-based strategies
(using a strategy pattern or config enum).
   - [ ] Add a boolean flag like `enable_constraint_check`.
   - [ ] Track feasible solution ratio per generation as a debug/diagnostic
metric.
*/

namespace Algorithm {
class IIndividual {
 public:
  virtual ~IIndividual() = default;

  virtual int fitness() const = 0;

  virtual void mutate() = 0;

  virtual void print() const = 0;

  virtual std::shared_ptr<IIndividual> clone() const = 0;

  virtual std::shared_ptr<IIndividual> crossover(
      const std::shared_ptr<IIndividual>& other) const = 0;
};

class GeneticAlgorithm {
 public:
  GeneticAlgorithm(int population_size, int generations, float mutation_rate,
                   float crossover_rate)
      : population_size(population_size),
        generations(generations),
        mutation_rate(mutation_rate),
        crossover_rate(crossover_rate) {
    srand(time(0));
  }

  // Initialize the population
  template <typename DerivedIndividual, typename... Args>
  auto initialize(Args&&... args) noexcept {
    population.clear();
    for (int i = 0; i < population_size; i++) {
      auto individual =
          std::make_shared<DerivedIndividual>(std::forward<Args>(args)...);
      population.push_back(individual);
    }
  }

  // Select an individual from the population using roulette wheel selection
  auto selection() const noexcept {
    // Calculate the total fitness of all individuals
    double totalFitness = 0.0;
    for (const auto& individual : population) {
      totalFitness += individual->fitness();
    }

    // Generate a random number between 0 and totalFitness
    double randValue = rand() / static_cast<double>(RAND_MAX) * totalFitness;

    // Select an individual based on the random value and cumulative fitness
    double cumulativeFitness = 0.0;
    for (const auto& individual : population) {
      cumulativeFitness += individual->fitness();
      if (cumulativeFitness >= randValue) {
        return individual;  // Select this individual
      }
    }

    // Default return the last individual (this is a safety check)
    return population.back();
  }

  // Run the genetic algorithm
  void run() noexcept {
    // Compare two individuals based on their fitness
    auto compare = [&](const auto& a, const auto& b) {
      return a->fitness() < b->fitness();
    };

    for (int j = 0; j < generations; j++) {
      decltype(population) new_population;

      for (int i = 0; i < population_size; i++) {
        auto parent1 = selection();
        auto parent2 = selection();

        decltype(parent1) child;
        if ((rand() % 100) < crossover_rate * 100) {
          child = parent1->crossover(parent2);
        } else {
          child = parent1->clone();
        }

        if ((rand() % 100) < mutation_rate * 100) {
          child->mutate();
        }
        new_population.emplace_back(child);
      }

      population = std::move(new_population);

      best_individual =
          *std::max_element(population.begin(), population.end(), compare);

      best_individual->print();

      if (best_individual->fitness() == 28) {
        break;
      }
    }
  }

 private:
  // The size of the population
  int population_size;

  // The number of generations
  int generations;

  // The mutation rate
  float mutation_rate;

  // The crossover rate
  float crossover_rate;

  // The population
  std::vector<std::shared_ptr<IIndividual>> population;

  // The best individual
  std::shared_ptr<IIndividual> best_individual;
};
}  // namespace Algorithm

#endif