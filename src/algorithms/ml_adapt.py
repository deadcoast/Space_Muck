"""
ml_adapt.py

Advanced machine learning & adaptation utilities for symbiote evolution and game AI.
Includes Reinforcement Learning (RL) agents, neuroevolution managers, Bayesian
uncertainty modeling, and persistent genome systems. Each component is designed to
plug into a broader game framework (e.g., SymbioteEvolutionAlgorithm) and manipulate
or store data about symbiote behaviors.

No usage examples are provided per instructions. The docstrings explain each class’s
methods and how to integrate them into the larger application.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------
1. Reinforcement Learning (SymbioteQLearner)
   - A tabular Q-learning agent for deciding high-level actions: “attack,” “grow,” etc.
   - Stores states (e.g., aggression level, population, resources), and chooses
     actions that maximize expected long-term reward.

2. Neuroevolution (NeuroEvolutionManager)
   - Evolves small neural networks for controlling symbiote behaviors.
   - Uses genetic or differential evolution to mutate and cross over neural network
     weights, enabling emergent strategies as the symbiotes adapt to the player.

3. Bayesian Updates for Uncertainty (BayesianSymbioteBelief)
   - Maintains a belief distribution about unknown variables (e.g., player fleet size).
   - Updates belief with new observations (e.g., partial intel or scouting).
   - Can be extended with Kalman filters, particle filters, or Bayesian updates to drive
     adaptive behavior under uncertainty.

4. Persistent Population Genomes (PopulationGenomeManager)
   - Manages storing and loading “genome” or ML state across multiple game sessions.
   - Over multiple playthroughs, symbiotes become increasingly adept at countering
     frequent player strategies.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025 ...
All rights reserved.
"""
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable

# Create a NumPy random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)


# -----------------------------------------------------------------------------
# 1) REINFORCEMENT LEARNING: SymbioteQLearner
# -----------------------------------------------------------------------------
class SymbioteQLearner:
    """
    A tabular Q-learning agent for high-level symbiote actions like "ATTACK",
    "GROW", "HARASS", etc. The user is responsible for:
      1. Defining the state representation (e.g. aggression, population, resources).
      2. Defining the reward function (passed to update_q()) each turn.
      3. Calling get_action(state) in the game loop to pick the agent's move.

    Integrate this agent into your existing code by:
      - Storing the agent in your SymbioteEvolutionAlgorithm or AI manager.
      - Each turn, compute a "state" and call get_action().
      - After seeing the result, call update_q() with the reward for that action.

    You can extend this to more advanced RL approaches if the state/action
    space grows, or if you want function approximation.
    """

    def __init__(
        self,
        actions: List[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        """
        Args:
            actions: The list of possible actions (strings).
            alpha: Learning rate for Q-learning.
            gamma: Discount factor for future rewards.
            epsilon: Exploration rate for epsilon-greedy action selection.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Dictionary structure: {(state_repr, action): q_value}
        self.q_table: Dict[Tuple[Any, str], float] = {}

    def get_action(self, state: Any) -> str:
        """
        Epsilon-greedy selection from Q-table. If random < epsilon, pick a
        random action; else pick the best known action from the Q-values.

        Args:
            state: Any hashable state representation.

        Returns:
            action: The chosen action from self.actions.
        """
        if rng.random() < self.epsilon:
            return rng.choice(self.actions)
        # Exploit best known Q-value
        best_action = None
        best_q = float("-inf")
        for a in self.actions:
            q = self.q_table.get((state, a), 0.0)
            if q > best_q:
                best_q = q
                best_action = a
        return rng.choice(self.actions) if best_action is None else best_action

    def update_q(
        self, old_state: Any, action: str, reward: float, new_state: Any
    ) -> None:
        """
        Standard Q-learning update: Q(s,a) ← Q(s,a) + alpha * [R + gamma * max(Q(s',·)) - Q(s,a)].

        Args:
            old_state: The previous state.
            action: The action taken.
            reward: The reward obtained.
            new_state: The resultant state after taking 'action'.
        """
        old_q = self.q_table.get((old_state, action), 0.0)
        max_future_q = float("-inf")
        for a in self.actions:
            candidate = self.q_table.get((new_state, a), 0.0)
            if candidate > max_future_q:
                max_future_q = candidate
        if max_future_q == float("-inf"):
            # No knowledge about new_state yet
            max_future_q = 0.0
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(old_state, action)] = new_q

    def adjust_epsilon(self, performance: float) -> None:
        """
        Adjusts exploration rate based on some performance measure.
        For instance, if agent is performing well, reduce exploration;
        if performing poorly, increase it.

        Args:
            performance: A value between 0 and 1 indicating how well the agent is doing.
        """
        if performance > 0.7:
            self.epsilon = max(0.01, self.epsilon * 0.95)
        elif performance < 0.3:
            self.epsilon = min(0.5, self.epsilon * 1.05)

    def save_q_table(self, filepath: str) -> None:
        """
        Saves the Q-table to disk using pickle.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filepath: str) -> None:
        """
        Loads a Q-table from disk, overwriting the current table.
        """
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)


# -----------------------------------------------------------------------------
# 2) NEUROEVOLUTION: NeuroEvolutionManager
# -----------------------------------------------------------------------------
class SimpleNeuralNetwork:
    """
    A minimal feedforward neural network with a single hidden layer for demonstration.
    The user can expand or restructure this to suit the complexity needed.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim: Number of input nodes.
            hidden_dim: Number of hidden nodes in one hidden layer.
            output_dim: Number of output nodes (e.g., action values).
        """
        # Xavier / He initialization can be used; for simplicity, random uniform
        limit_in = 1.0 / max(1, input_dim)
        limit_hid = 1.0 / max(1, hidden_dim)

        # Weights: input->hidden
        self.w1 = rng.uniform(-limit_in, limit_in, (input_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,))
        # Weights: hidden->output
        self.w2 = rng.uniform(-limit_hid, limit_hid, (hidden_dim, output_dim))
        self.b2 = np.zeros((output_dim,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the network. Using ReLU for hidden activation, or can
        switch to sigmoid/tanh. Output is linear or you can apply softmax.
        Args:
            x: shape (input_dim,)
        Returns:
            out: shape (output_dim,)
        """
        z1 = x.dot(self.w1) + self.b1
        h1 = np.maximum(0, z1)  # ReLU
        return h1.dot(self.w2) + self.b2

    def clone(self) -> "SimpleNeuralNetwork":
        """
        Creates a deep copy of this network. Used for certain evolution steps.
        """
        clone_net = SimpleNeuralNetwork(1, 1, 1)  # Dummy dims
        clone_net.w1 = self.w1.copy()
        clone_net.b1 = self.b1.copy()
        clone_net.w2 = self.w2.copy()
        clone_net.b2 = self.b2.copy()
        clone_net.__dict__["w1"] = self.w1.copy()
        clone_net.__dict__["b1"] = self.b1.copy()
        clone_net.__dict__["w2"] = self.w2.copy()
        clone_net.__dict__["b2"] = self.b2.copy()
        # Overwrite dims from original
        clone_net.__dict__["w1"].shape = self.w1.shape
        clone_net.__dict__["w2"].shape = self.w2.shape
        clone_net.__dict__["b1"].shape = self.b1.shape
        clone_net.__dict__["b2"].shape = self.b2.shape
        return clone_net


class NeuroEvolutionManager:
    """
    Manages a population of neural networks, each representing a potential symbiote
    decision-making or strategy controller. Uses a simple genetic algorithm for
    selection, crossover, and mutation. Over time, the fittest networks survive,
    leading to emergent behaviors.

    Common usage:
       1. Initialize with a population size, specifying the neural net architecture.
       2. Evaluate each network in a simulation or partial scenario, collecting fitness.
       3. Call evolve() to produce a new generation from the best performers.
       4. Repeat until converged or until you want to deploy the best network.

    Integration:
       - Each symbiote colony could have an associated network controlling aggression
         or feeding decisions. After each round, you measure success (fitness).
         The best networks replicate and mutate, the worst are replaced.
    """

    def __init__(
        self, population_size: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        """
        Args:
            population_size: How many networks to maintain at once.
            input_dim, hidden_dim, output_dim: Architecture definition.
        """
        self.population_size = population_size
        self.networks: List[SimpleNeuralNetwork] = []
        self.fitness_scores: List[float] = []

        for _ in range(population_size):
            net = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim)
            self.networks.append(net)
        self.fitness_scores = [0.0] * population_size

        # E.g. evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5

    def evaluate_fitness(self, index: int, fitness: float) -> None:
        """
        Assign or accumulate a fitness score to the network at population index.
        Usually called after simulating that network in a test scenario.
        """
        self.fitness_scores[index] = fitness

    def evolve(self) -> None:
        """
        Basic genetic evolution of the population:
          1. Sort by fitness.
          2. Keep top 50% as parents.
          3. Generate offspring by crossover and mutation.
        """
        # Sort networks by fitness descending
        indices_sorted = sorted(
            range(self.population_size),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )
        half = self.population_size // 2
        parents_idx = indices_sorted[:half]
        # The new population will be built from these parents
        new_population: List[SimpleNeuralNetwork] = [
            self.networks[parents_idx[0]].clone()
        ]

        if half > 1:
            new_population.append(self.networks[parents_idx[1]].clone())

        # Fill the rest
        while len(new_population) < self.population_size:
            # Select two parents from parents_idx
            p1 = rng.choice(parents_idx)
            p2 = rng.choice(parents_idx)
            child_net = self._crossover(self.networks[p1], self.networks[p2])
            self._mutate(child_net)
            new_population.append(child_net)

        self.networks = new_population
        self.fitness_scores = [0.0] * self.population_size

    def _crossover(
        self, net_a: SimpleNeuralNetwork, net_b: SimpleNeuralNetwork
    ) -> SimpleNeuralNetwork:
        """
        Single-point or uniform crossover of net_a and net_b weights.
        Returns a new child network.
        """
        child = net_a.clone()
        if rng.random() < self.crossover_rate:
            # Example: crossover in hidden layer weights
            shape_w1 = net_a.w1.shape
            shape_w2 = net_a.w2.shape
            # Flatten weights for uniform crossover
            flat_a_w1 = net_a.w1.flatten()
            flat_b_w1 = net_b.w1.flatten()
            flat_a_w2 = net_a.w2.flatten()
            flat_b_w2 = net_b.w2.flatten()

            new_w1 = self._create_combined_weight_matrix(flat_a_w1, flat_b_w1, shape_w1)
            new_w2 = self._create_combined_weight_matrix(flat_a_w2, flat_b_w2, shape_w2)
            child.w1 = new_w1
            child.w2 = new_w2

            # Same for biases if desired
            for i in range(len(child.b1)):
                child.b1[i] = net_a.b1[i] if rng.random() < 0.5 else net_b.b1[i]
            for i in range(len(child.b2)):
                child.b2[i] = net_a.b2[i] if rng.random() < 0.5 else net_b.b2[i]

        return child

    def _create_combined_weight_matrix(self, weights_a, weights_b, shape):
        """Create a new weight matrix by randomly combining values from two parent networks.
        
        Args:
            weights_a: Flattened weights from first parent network
            weights_b: Flattened weights from second parent network
            shape: Shape to reshape the resulting weights array into
            
        Returns:
            Combined weight matrix with parent values randomly selected for each position
        """
        # For each dimension, pick from parent A or B
        result = []
        for i in range(len(weights_a)):
            if rng.random() < 0.5:
                result.append(weights_a[i])
            else:
                result.append(weights_b[i])
        return np.array(result).reshape(shape)

    def _mutate(self, net: SimpleNeuralNetwork) -> None:
        """
        Adds random perturbations to the child's weights & biases
        with probability = self.mutation_rate.
        """
        w1_flat = net.w1.flatten()
        for i in range(len(w1_flat)):
            if rng.random() < self.mutation_rate:
                w1_flat[i] += rng.normal(0, 0.1)
        net.w1 = w1_flat.reshape(net.w1.shape)

        w2_flat = net.w2.flatten()
        for i in range(len(w2_flat)):
            if rng.random() < self.mutation_rate:
                w2_flat[i] += rng.normal(0, 0.1)
        net.w2 = w2_flat.reshape(net.w2.shape)

        # Similarly mutate biases
        for i in range(len(net.b1)):
            if rng.random() < self.mutation_rate:
                net.b1[i] += rng.normal(0, 0.1)
        for i in range(len(net.b2)):
            if rng.random() < self.mutation_rate:
                net.b2[i] += rng.normal(0, 0.1)


# -----------------------------------------------------------------------------
# 3) BAYESIAN UPDATES FOR UNCERTAINTY: BayesianSymbioteBelief
# -----------------------------------------------------------------------------
class BayesianSymbioteBelief:
    """
    Maintains a belief distribution about unknown or partially observed variables,
    such as the player's fleet size or resource stockpile. Each time new evidence
    is gained (e.g., scouting info, partial intel from an attack), the distribution
    is updated using Bayesian or approximate filtering.

    For computational efficiency, you can discretize the variable domain (e.g., possible
    fleet sizes from 0 to 100) and store a probability distribution. On each update,
    incorporate the likelihood of new observations, re-normalize, etc.

    Extend or modify to handle more advanced modeling: multiple variables, dependencies,
    or a Kalman filter if continuous.
    """

    def __init__(
        self,
        variable_name: str,
        domain_size: int,
        prior_distribution: Optional[List[float]] = None,
    ):
        """
        Args:
            variable_name: A label for the unknown quantity (e.g. "player_fleet").
            domain_size: The number of discrete states in the domain.
            prior_distribution: Optional initial distribution over the domain.
        """
        self.variable_name = variable_name
        self.domain_size = domain_size

        if prior_distribution is None:
            # Default to uniform
            self.belief = [1.0 / domain_size] * domain_size

        elif len(prior_distribution) != domain_size:
            raise ValueError("Prior distribution length mismatch with domain_size.")
        else:
            self.belief = [float(p) for p in prior_distribution]

    def update_belief(self, likelihood_func: Callable[[int], float]) -> None:
        """
        Perform a Bayesian update with a user-supplied likelihood function that
        returns P(observation | state). For each possible state s in [0, domain_size),
        we multiply the old belief by likelihood_func(s) and then re-normalize.

        Args:
            likelihood_func: A function that takes a discrete state index and
                             returns a likelihood for the current observation.
        """
        new_belief = [
            self.belief[s] * likelihood_func(s) for s in range(self.domain_size)
        ]
        # Normalize
        total = sum(new_belief)
        if total > 0:
            self.belief = [val / total for val in new_belief]

    def most_likely_state(self) -> int:
        """
        Returns the index of the state with the highest posterior probability.
        """
        return int(np.argmax(self.belief))

    def expected_value(self) -> float:
        """
        Returns the expected value of the variable if states are in [0..domain_size-1].
        """
        # Weighted average
        ev = 0.0
        for s in range(self.domain_size):
            ev += s * self.belief[s]
        return ev


# -----------------------------------------------------------------------------
# 4) PERSISTENT POPULATION GENOMES
# -----------------------------------------------------------------------------
class PopulationGenomeManager:
    """
    Stores or loads a "genome" representing a population’s learned parameters
    or ML state across multiple game sessions. This is crucial for persistent
    evolution: over many playthroughs, the symbiotes adapt more aggressively
    to common player tactics.

    You might store:
      - Q-tables,
      - Evolved neural network weights,
      - Bayesian distributions,
      - or a custom dictionary of relevant parameters.

    On game start, load_genome() to retrieve the last known state. On game end,
    save_genome() to store any updated changes. Over time, this yields an
    evolving meta-game experience.
    """

    def __init__(self, filepath: str):
        """
        Args:
            filepath: The path where the genome state will be saved/loaded.
        """
        self.filepath = filepath
        # Store arbitrary data in a dictionary: "q_table", "best_network", "bayes_belief", ...
        self.genome_data: Dict[str, Any] = {}

    def set_data(self, key: str, value: Any) -> None:
        """
        Insert or replace data in the genome under a given key.
        """
        self.genome_data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from the genome. Returns 'default' if key not found.
        """
        return self.genome_data.get(key, default)

    def load_genome(self) -> None:
        """
        Loads the genome from self.filepath, overwriting the current genome_data.
        If no file is found or loading fails, keeps the existing data.
        """
        try:
            with open(self.filepath, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self.genome_data = loaded
        except FileNotFoundError:
            pass  # No prior genome found
        except Exception as e:
            print(f"Error loading genome from {self.filepath}: {e}")

    def save_genome(self) -> None:
        """
        Saves the current genome_data to disk at self.filepath using pickle.
        """
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(self.genome_data, f)
        except Exception as e:
            print(f"Error saving genome to {self.filepath}: {e}")
