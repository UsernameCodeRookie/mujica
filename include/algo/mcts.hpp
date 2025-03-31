#ifndef GENETIC_HPP
#define GENETIC_HPP

#include <algorithm>
#include <cmath>

#include "dnn/group.hpp"

namespace Algorithm {
class IState {
 public:
  // Take a random action and return the next state
  virtual std::shared_ptr<IState> takeAction(bool random) = 0;

  // Check if all actions are expanded
  virtual bool isAllExpanded() const = 0;

  // Check if the state is terminated
  virtual bool isTerminated() const = 0;

  // Evaluate the state
  virtual int evaluate() const = 0;

  virtual void print() const = 0;
};

class Node {
 public:
  Node(const std::shared_ptr<IState> _state,
       const std::shared_ptr<Node> _parent = nullptr)
      : state(_state), parent(_parent) {}

  // Get state
  auto getState() const noexcept { return state; }

  // Get parent node
  auto getParent() const noexcept { return parent; }

  // Set parent node
  void setParent(const std::shared_ptr<Node> _parent) { parent = _parent; }

  // Get children nodes
  auto getChildren() const noexcept { return children; }

  // Add child node
  void addChild(const std::shared_ptr<Node> child) {
    children.push_back(child);
  }

  int visits;

  int reward;

 private:
  // State
  std::shared_ptr<IState> state;

  // Parent node
  std::shared_ptr<Node> parent;

  // Children nodes
  std::vector<std::shared_ptr<Node>> children;
};

class MonteCarloTreeSearch {
 public:
  MonteCarloTreeSearch(int _budget, std::shared_ptr<Node> _root, bool _random)
      : budget(_budget), root(_root), random(_random) {}

  // Select the child with the highest value
  auto select(std::shared_ptr<Node> node) const noexcept {
    float best_score = -std::numeric_limits<float>::infinity();
    std::shared_ptr<Node> best_node;

    for (auto child : node->getChildren()) {
      float avg = child->reward / child->visits;
      float score = avg + std::sqrt(2 * std::log(node->visits) / child->visits);

      if (score > best_score) {
        best_score = score;
        best_node = child;
      }
    }
    return best_node;
  }

  // Expand the node by taking a random action
  auto expand(const std::shared_ptr<Node> node) const noexcept {
    auto state = node->getState();
    state = state->takeAction(random);

    // Make new node
    auto sub_node = std::make_shared<Node>(state, node);
    node->addChild(sub_node);

    return sub_node;
  }

  // Tree policy
  auto treePolicy(std::shared_ptr<Node> node) const noexcept {
    // Step while the state is not terminal
    while (!node->getState()->isTerminated()) {
      // If the node is not expanded, expand it
      if (!node->getState()->isAllExpanded()) return expand(node);

      // Select the child with the highest value
      node = select(node);
    }
    return node;
  }

  // Default policy
  auto defaultPolicy(const std::shared_ptr<Node> node) const noexcept {
    // Get the current state
    auto state = node->getState();

    // Simulate the game until the end
    while (!state->isTerminated()) {
      // Take random action
      state = state->takeAction(random);
    }

    int reward = state->evaluate();

    return reward;
  }

  // Backpropagate the result
  void backPropagate(int reward, std::shared_ptr<Node> node) const noexcept {
    // Backpropagate the result
    while (node != nullptr) {
      // Update the number of visits
      node->visits++;

      // Update the reward
      node->reward += reward;

      // Get the parent node
      node = node->getParent();
    }
  }

  // Search
  auto search() const noexcept {
    // Run computation budget times
    for (int _ = 0; _ < budget; _++) {
      // 1. Select or create a leaf node from the nodes already contained
      // within the search tree
      auto expand_node = treePolicy(root);

      // 2.Play out the domain from a given nonterminal state to produce a
      // value estimate
      int reward = defaultPolicy(expand_node);

      // 3. Updates node statistics that inform future tree policy decisions.
      backPropagate(reward, expand_node);
    }
  }

 private:
  // Computation budget
  int budget;

  // Root node
  std::shared_ptr<Node> root;

  // Random
  bool random;
};

}  // namespace Algorithm
#endif