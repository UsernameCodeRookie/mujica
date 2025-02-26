#ifndef GENETIC_HPP
#define GENETIC_HPP

#include <algorithm>

#include "dnn/group.hpp"

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
  template <typename DerivedIndividual>
  auto initialize() noexcept {
    population.clear();
    for (int i = 0; i < population_size; i++) {
      auto individual = std::make_shared<DerivedIndividual>();
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

      auto best_individual =
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
};

#endif