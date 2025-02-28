#ifndef ANNEALING_HPP
#define ANNEALING_HPP

#include <cmath>

#include "dnn/group.hpp"

namespace Algorithm {
class IState {
 public:
  virtual ~IState() {}

  // Calculate the energy (objective function value) for a given solution
  virtual int evaluate() const = 0;

  // Generate a neighboring solution for a given current solution
  virtual std::shared_ptr<IState> getNeighbor() const = 0;

  // Print the current state
  virtual void print() const = 0;
};

class SimulatedAnnealing {
 public:
  SimulatedAnnealing(std::shared_ptr<IState> state, double initial_temperature,
                     double min_temperature, double cooling_rate)
      : state(state),
        initial_temperature(initial_temperature),
        min_temperature(min_temperature),
        cooling_rate(cooling_rate) {
    srand(time(0));
  }

  // Run the simulated annealing algorithm
  void run() const noexcept {
    // Initialize the solution with a random value in the range [-1, 1]
    auto current_solution = state;
    int current_energy = state->evaluate();
    double temperature = initial_temperature;

    // Main loop for the simulated annealing process
    while (temperature >= min_temperature) {
      // Generate a neighboring solution
      auto new_solution = state->getNeighbor();
      int new_energy = state->evaluate();

      // Decide whether to accept the new solution based on its energy and
      // temperature
      if (acceptSolution(current_energy, new_energy)) {
        current_solution = new_solution;
        current_energy = new_energy;
        printf("Temperature: %f, Energy: %d\n", temperature, current_energy);
      }

      // Decrease the temperature for the next iteration
      temperature *= cooling_rate;
    }

    current_solution->print();
  }

 private:
  // Decide whether to accept the new solution based on its energy and
  // temperature
  bool acceptSolution(double current_energy, double new_energy) const noexcept {
    if (new_energy < current_energy) {
      return true;  // Accept the new solution if it's better
    }
    // If the new solution is worse, accept it with a probability based on the
    // temperature
    double acceptance_prob =
        exp((current_energy - new_energy) / initial_temperature);
    return ((rand() % 1000) / 1000.0) < acceptance_prob;
  }

  // Pointer to the external state implementation
  std::shared_ptr<IState> state;

  // Initial temperature for the simulated annealing process
  double initial_temperature;

  // Minimum temperature for the simulated annealing process
  double min_temperature;

  // Rate at which the temperature decreases
  double cooling_rate;
};
}  // namespace Algorithm

#endif  // ANNEALING_HPP