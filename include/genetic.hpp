#ifndef GENETIC_HPP
#define GENETIC_HPP

#include <algorithm>

#include "dnn/group.hpp"

class Individual {
 public:
  virtual ~Individual() = default;
  virtual int fitness() const = 0;
  virtual void mutate() = 0;
  virtual void print() const = 0;
};

class GA {
 public:
  GA(int population_size, int generations, float mutation_rate)
      : population_size(population_size),
        generations(generations),
        mutation_rate(mutation_rate) {
    srand(time(0));
  }

  // Initialize the population
  auto initialize_population() const noexcept {
    auto population = std::vector<std::shared_ptr<Individual>>();
    for (int i = 0; i < population_size; i++) {
      auto individual = std::make_shared<Individual>();
      population.push_back(individual);
    }
    return population;
  }

  // Select an individual from the population
  auto selection(const std::vector<std::shared_ptr<Individual>>& population)
      const noexcept {
    int total_fitness = 0;
    for (const auto& individual : population) {
      total_fitness += individual->fitness();
    }

    int pick = rand() % total_fitness;
    int current = 0;
    for (const auto& individual : population) {
      current += individual->fitness();
      if (current > pick) {
        return individual;
      }
    }
    return population.back();
  }

  // Run the genetic algorithm
  void run() const noexcept {
    auto population = initialize_population();

    for (int _ = 0; _ < generations; _++) {
      // Create a new population
      decltype(population) new_population;

      while (new_population.size() < population_size) {
        auto child = selection(population);

        if ((rand() % 100) < mutation_rate * 100) {
          child->mutate();
        }

        new_population.push_back(child);
      }

      population = new_population;

      auto compare = [&](const auto& a, const auto& b) {
        return a->fitness() < b->fitness();
      };

      auto best_individual =
          *std::max_element(population.begin(), population.end(), compare);

      best_individual->print();
    }
  }

 private:
  // The size of the population
  int population_size;

  // The number of generations
  int generations;

  // The mutation rate
  float mutation_rate;
};

#endif