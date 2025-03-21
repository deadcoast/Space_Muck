"""
economy_decision.py

Provides advanced economic and decision-making systems, expanding the symbiote-player
resource interactions. This includes game-theoretic feeding decisions (especially for
multiplayer), variable mineral valuation that adapts to how often minerals are used,
and cost-benefit attack analysis for more realistic aggression.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------

1) MineralResourceOptimizer:
   - Uses scipy.optimize to solve a multi-objective or constrained optimization
     that balances selling minerals vs. feeding them, under potential constraints
     like "minimum feeding to avoid aggression" or "desired revenue for fleet upgrades."

2) PlayerFeedingBehaviorModel:
   - Applies scikit-learn (KMeans & LogisticRegression) to categorize or predict
     how players feed. This can let symbiotes tailor their aggression or deals
     to each player's style (e.g., "feeder" vs. "hoarder" cluster).

3) Matrix-based Payoff Analysis:
   - Demonstrates how to compute resource payoffs or partial equilibrium
     using NumPy arrays for multiple players/factions, with vectorized operations.

4) Game-TheoreticFeedingModel
   - Allows symbiotes to react strategically to how often they are fed or starved.
   - In a multiplayer context, can identify which players feed frequently vs. hoard
     minerals, then punish or reward them accordingly.
   - Implements a mini "payoff matrix" approach or repeated game strategy for
     docile vs. aggressive responses.

5) VariableMineralValuation
   - Dynamically adjusts the effective "value" or potency of each mineral type (rare,
     precious, anomaly, etc.) based on usage frequency or ML signals.
   - As certain minerals get overused, the symbiotes might develop resistance (lowered
     effect) or become more addicted (raised effect), creating a meta-economy around
     feeding strategies.

6) SymbioteAttackEVCalculator
   - Considers a cost-benefit approach to deciding attacks, computing an expected
     value of launching an attack vs. waiting or growing first.
   - Integrates with RL or evolutionary logic by providing a numerical payoff that
     the AI can optimize.
   - EV(attack) = P(victory)*spoils - P(defeat)*losses. Compare to EV(wait).

All classes here are production-ready building blocks. No usage examples are provided
per instructions. Integrate them by creating instances and calling their methods in
your main game loop, hooking into your existing ML or aggression systems.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025 ...
All rights reserved.
"""

import contextlib

# Standard library imports
import importlib.util
import warnings
from typing import Dict, List

# Advanced scientific and ML libraries
import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Local application imports - import only what we need
from config import (  # Colors for mineral types and visualization; UI colors for data visualization; Entity behavior colors
    COLOR_ASTEROID_ANOMALY,
    COLOR_ASTEROID_PRECIOUS,
    COLOR_ASTEROID_RARE,
    COLOR_ENTITY_FEEDING,
    COLOR_ERROR,
    COLOR_GRID,
    COLOR_SUCCESS,
    COLOR_UI_BG,
    COLOR_UI_BORDER,
    COLOR_UI_TEXT,
    COLOR_WARNING,
)

# No need for availability flags - we're always using the advanced libraries

# Check if pygame is available for visualization
PYGAME_AVAILABLE = importlib.util.find_spec("pygame") is not None
if PYGAME_AVAILABLE:
    import pygame

# Import numpy - this is required, not optional
try:
    import numpy as np
except ImportError:
    warnings.warn("NumPy is required for economy_decision.py functionality.")

# No need for conditional imports - we're always using the advanced libraries

# Import optional pygame
if PYGAME_AVAILABLE:
    with contextlib.suppress(ImportError):
        import pygame


# -----------------------------------------------------------------------------
# 1) MINERAL RESOURCE OPTIMIZER (SCIPY OPTIMIZE)
# -----------------------------------------------------------------------------
class MineralResourceOptimizer:
    """
    Formulates a multi-objective or constrained optimization problem:
      Decide how many minerals to feed vs. sell to maximize some combination
      of (player profit, symbiote appeasement, future growth).
    Potential constraints:
      - At least 'X' feeding to keep aggression below threshold
      - At most 'Y' total minerals can be allocated
      - Possibly other constraints (like keep at least some buffer).

    This uses scipy.optimize (linear program or nonlinear) to find an optimum.
    If scipy is not available, falls back to a simpler greedy algorithm.

    Example structure:
      We define variables:
        x_c = how many common minerals to feed
        x_r = how many rare minerals to feed
        x_p = how many precious minerals to feed
        ...
      Then we have constraints and an objective function.
    """

    def __init__(self, mineral_types: List[str]):
        """
        Args:
            mineral_types: A list of recognized mineral types (e.g., ["common","rare","precious"]).
        """
        self.mineral_types = mineral_types

        # Colors for visualization based on mineral type
        self.mineral_colors = {
            "common": COLOR_GRID,  # Common minerals
            "rare": COLOR_ASTEROID_RARE,  # Rare minerals
            "precious": COLOR_ASTEROID_PRECIOUS,  # Precious minerals
            "anomaly": COLOR_ASTEROID_ANOMALY,  # Anomaly minerals
        }
        # Status colors for UI feedback
        self.status_colors = {
            "optimal": COLOR_SUCCESS,
            "fallback": COLOR_WARNING,
            "error": COLOR_ERROR,
        }

    def linear_optimize_feeding(
        self,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None,
        aggression_coef: float = 0.0,
    ) -> Dict[str, float]:
        """
        Solve a linear optimization where we aim to maximize:
            Sum( feed_benefits[m] * feed[m] ) + Sum( sell_values[m] * (total_minerals[m] - feed[m]) )
            - aggression_coef * (aggression if we feed below some threshold)

        This is a toy example that uses linear (and maybe piecewise) constraints.

        Args:
            total_minerals: e.g. {"common": 30, "rare": 10, "precious": 5}
            feed_benefits: e.g. {"common": 1.0, "rare": 3.0, "precious": 5.0}
                -> how much "value" or "appeasement" gained from feeding each type
            sell_values: e.g. {"common": 2.0, "rare": 10.0, "precious": 20.0}
            min_feeding_requirements: e.g. {"common": 5}, means we must feed at least 5 common minerals
            max_aggression: if we want to incorporate a constraint or penalty for aggression
            aggression_coef: how severely shortfalls in feeding hamper the objective.

        Returns:
            A dict of how many of each mineral type we feed (the rest are sold).
        """
        # For demonstration, we do a basic linear program with the format:
        # maximize c^T x
        # subject to Ax <= b, x >= 0
        # x = feed amounts for each mineral.

        # We transform it into minimize -c^T x if using scipy's linear_sum_assignment approach
        # or the newer `linprog` from SciPy. Let's do a quick example with `linprog`.

        # Build objective vector: for each mineral type, objective coefficient = feed_benefits[m] + sell gain * ???
        # Actually, if we feed x[m], we lose the chance to sell x[m]. Let's define variables carefully.

        # We'll do a simple approach:
        # max( sum_m (feed_benefits[m] * x_m) + sum_m(sell_values[m]*(total_m[m]-x_m)) )
        #    = sum_m(sell_values[m]*total_m[m]) + sum_m( (feed_benefits[m] - sell_values[m]) * x_m )
        # We can ignore the constant sum_m(sell_values[m]*total_m[m]) for maximizing => doesn't affect x
        # => objective is sum_m( (feed_benefits[m] - sell_values[m]) * x_m )

        # Then constraints: 0 <= x_m <= total_m[m], plus min feeding if desired.

        n = len(self.mineral_types)
        c = [-(feed_benefits[m] - sell_values[m]) for m in self.mineral_types]
        # Build constraints
        # We have x_i <= total_m[m]
        # Possibly we also incorporate min feeding, x_i >= minFeeding => that's negative constraints

        # A_ub * x <= b_ub
        a_ub = []
        b_ub = []
        # for x_i <= total_m[m], we do +1 on that variable
        # constraints: x_i - total_m[m] <= 0
        for i, m in enumerate(self.mineral_types):
            row = [0] * n
            row[i] = 1
            a_ub.append(row)
            b_ub.append(total_minerals[m])

        # If we want min_feeding_requirements, we do x_i >= min => -x_i >= -min
        # => -x_i <= -min => A_ub row with -1 for x_i
        if min_feeding_requirements:
            for i, m in enumerate(self.mineral_types):
                if m in min_feeding_requirements:
                    row = [0] * n
                    row[i] = -1
                    a_ub.append(row)
                    b_ub.append(-min_feeding_requirements[m])

        bounds = [(0, None) for _ in self.mineral_types]
        a_ub = np.array(a_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)

        # Solve using SciPy's advanced optimization
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")

        # Always use the optimization result - no fallbacks
        solution = res.x
        return {m: float(solution[i]) for i, m in enumerate(self.mineral_types)}

    # -----------------------------------------------------------------------------
    # 2) PLAYER FEEDING BEHAVIOR MODEL (scikit-learn)
    # -----------------------------------------------------------------------------
    def _fallback_proportional_feeding(
        self,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """
        Simple fallback when linear optimization is unavailable or fails.
        Feeds minerals proportionally to their benefit, up to meeting minimum requirements.

        Args:
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            min_feeding_requirements: Minimum feeding requirements by mineral type

        Returns:
            Dictionary with {mineral_type: amount_to_feed}
        """
        # First, handle any explicit minimum feeding requirements
        result = {m: 0.0 for m in self.mineral_types}
        remaining_minerals = total_minerals.copy()

        if min_feeding_requirements:
            for m, min_amount in min_feeding_requirements.items():
                if m in remaining_minerals:
                    # Feed the minimum required amount, up to what's available
                    feed_amount = min(min_amount, remaining_minerals[m])
                    result[m] = feed_amount
                    remaining_minerals[m] -= feed_amount

        # For the remaining minerals, allocate proportionally based on feed benefits
        # Calculate total benefit weighted by amount available
        total_weighted_benefit = sum(
            feed_benefits.get(m, 0) * amt
            for m, amt in remaining_minerals.items()
            if amt > 0
        )

        if total_weighted_benefit > 0:
            for m, remaining in remaining_minerals.items():
                if remaining > 0 and m in feed_benefits:
                    # Allocate based on relative benefit proportion
                    benefit_proportion = feed_benefits[m] / sum(feed_benefits.values())
                    feed_amount = remaining * benefit_proportion
                    result[m] += feed_amount

        return result

    def visualize_decision(self, surface, rect, total_minerals, feeding_amounts):
        """
        Visualize the feeding vs. selling decision on a pygame surface.

        Args:
            surface: Pygame surface to draw on
            rect: Rectangle area to draw within (x, y, width, height)
            total_minerals: Dictionary of {mineral_type: total_amount}
            feeding_amounts: Dictionary of {mineral_type: amount_to_feed}

        This creates a visual representation showing what proportion of each
        mineral type is being fed vs. sold.
        """
        if not PYGAME_AVAILABLE or pygame is None:
            return

        # Calculate selling amounts (total - feeding)
        selling_amounts = {}
        for mineral_type, total in total_minerals.items():
            feeding = feeding_amounts.get(mineral_type, 0)
            selling_amounts[mineral_type] = total - feeding

        # Setup drawing area
        x, y, width, height = rect
        bar_height = height / max(len(total_minerals), 1)
        padding = bar_height * 0.1

        # Draw mineral allocation bars
        current_y = y
        font = pygame.font.SysFont("Arial", int(bar_height * 0.4))

        for i, mineral_type in enumerate(self.mineral_types):
            if mineral_type not in total_minerals or total_minerals[mineral_type] <= 0:
                continue

            total = total_minerals[mineral_type]
            # Calculate proportions
            feeding = feeding_amounts.get(mineral_type, 0)
            selling = selling_amounts.get(mineral_type, 0)

            feed_width = (feeding / total) * width if total > 0 else 0
            sell_width = (selling / total) * width if total > 0 else 0

            # Get colors - fallback to default colors if type not found
            mineral_color = self.mineral_colors.get(mineral_type, COLOR_UI_TEXT)

            # Draw background
            pygame.draw.rect(
                surface, COLOR_UI_BG, (x, current_y, width, bar_height - padding)
            )

            # Draw feeding portion
            if feeding > 0:
                pygame.draw.rect(
                    surface,
                    COLOR_ENTITY_FEEDING,
                    (x, current_y, feed_width, bar_height - padding),
                )

            # Draw selling portion
            if selling > 0:
                pygame.draw.rect(
                    surface,
                    COLOR_SUCCESS,
                    (x + feed_width, current_y, sell_width, bar_height - padding),
                )

            # Draw border
            pygame.draw.rect(
                surface, COLOR_UI_BORDER, (x, current_y, width, bar_height - padding), 1
            )

            # Add text label
            text = font.render(
                f"{mineral_type}: {int(feeding)}/{int(total)}", True, mineral_color
            )
            surface.blit(text, (x + 5, current_y + (bar_height - padding) / 3))

            current_y += bar_height


class PlayerFeedingBehaviorModel:
    """
    Uses scikit-learn to cluster or classify players based on feeding logs
    and other behaviors. This might let symbiotes or the game system adapt
    to player “profiles” (e.g. “risk-averse feeder,” “aggressive hoarder,” etc.)
    """

    def __init__(self, n_clusters: int = 3):
        """
        We can keep both a KMeans for unsupervised clustering, plus
        a logistic reg for supervised prediction if we have labeled data
        (like "did the player feed next turn or not?").
        """
        self.n_clusters = n_clusters
        self.cluster_labels = None

        # Always initialize scikit-learn models - no conditionals
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.logistic_model = LogisticRegression(random_state=42, max_iter=200)

    def cluster_players(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Clusters players into n_clusters categories using their feeding patterns
        or other economic features.

        Args:
            feature_matrix: shape [n_players, n_features], e.g. each player's
                            average feed amount, variance, aggression, etc.

        Returns:
            A 1D array of cluster labels. Also stored in self.cluster_labels.
        """
        # Always use scikit-learn clustering - no fallbacks

        self.cluster_labels = self.kmeans_model.fit_predict(feature_matrix)
        return self.cluster_labels

    def _fallback_clustering(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Simple fallback clustering when scikit-learn is not available.
        Uses basic thresholding to group players.

        Args:
            feature_matrix: shape [n_players, n_features]

        Returns:
            A 1D array of cluster assignments
        """
        # Simple clustering based on average feature values
        if feature_matrix.size == 0:
            return np.array([])

        # Use mean values of features for thresholding
        means = np.mean(feature_matrix, axis=0)

        # Assign clusters based on whether features are above or below mean
        # This is a very simplified approach but works as a fallback
        cluster_labels = np.zeros(feature_matrix.shape[0], dtype=int)

        # For each player, check if their features are mostly above mean
        for i, player_features in enumerate(feature_matrix):
            # Count how many features are above mean
            above_mean = np.sum(player_features > means)
            # If more than half features are above mean, assign to cluster 1, else 0
            if above_mean > feature_matrix.shape[1] // 2:
                cluster_labels[i] = 1
            # If we want a third cluster for players close to the mean:
            elif np.all(np.abs(player_features - means) < 0.2 * means):
                cluster_labels[i] = 2

        self.cluster_labels = cluster_labels
        return cluster_labels

    def train_feed_prediction(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train a logistic regression to predict whether a player will feed next turn.

        Args:
            x_train: shape [n_samples, n_features], e.g. historical states
            y_train: shape [n_samples], binary or multi-class labels (feed yes/no)
        """
        # Always use scikit-learn's logistic regression model
        self.logistic_model.fit(x_train, y_train)

    def predict_feeding(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict binary feed vs. not feed for new examples.

        Args:
            x_test: shape [n_test, n_features]

        Returns:
            A 1D array of predictions (0 or 1).
        """
        # Always use scikit-learn's prediction
        return self.logistic_model.predict(x_test)

    def predict_feeding_proba(self, x_test: np.ndarray) -> np.ndarray:
        """
        Returns the probability of feeding (class 1) for each test sample.
        """
        # Always use scikit-learn's probability estimation
        return self.logistic_model.predict_proba(x_test)[:, 1]


# -----------------------------------------------------------------------------
# 3) MATRIX-BASED PAYOFF ANALYSIS
# -----------------------------------------------------------------------------
def compute_multi_faction_payoff(
    strategy_matrix: np.ndarray, payoff_tensor: np.ndarray
) -> np.ndarray:
    """
    Suppose we have M factions and K possible strategies each. strategy_matrix is
    a one-hot representation of each faction's chosen strategy, or a distribution
    over strategies. payoff_tensor is shape [M, K, K, ..., K], representing
    the payoff for each faction for each combination of chosen strategies.

    This function performs a vectorized approach to find the payoffs for all M
    factions given the distributions in strategy_matrix.

    Example:
      - If each faction chooses exactly 1 strategy (pure):
        strategy_matrix for faction i is a length-K vector with 1 at the chosen strategy
        index, 0 elsewhere.
      - payoff_tensor[i, s1, s2, ..., sM] is the payoff for faction i if
        faction0 uses s1, faction1 uses s2, ... faction(M-1) uses sM.

    We'll do a sum-product approach to handle mixed strategies as well:
      payoff_i = sum_{all combos} ( prob_of_that_combo * payoff_tensor[i, combo_of_strats] )

    For M factions, each has K possible strategies => K^M combos. We do a
    vectorized approach with np.einsum or similar.
    """
    # Note: This can get large quickly if M or K is big. For demonstration only.

    # strategy_matrix shape: [M, K]
    # Each row sums to 1 if it's a distribution over strategies for that faction.
    M, K = strategy_matrix.shape
    # payoff_tensor shape: [M, K, K, ..., K] (M+1 dims)
    # We'll compute expected payoff for each faction i by summation over all strategy combos
    # Weighted by the probability of that combo from the product of each faction's chosen distribution.

    # Build a grid of shape [K, K, ..., K] for the probabilities of each combo
    # Then multiply by payoff_tensor and sum. We can do np.einsum or manual expansions.

    # Probability of each combination is strategy_matrix[0, s0] * strategy_matrix[1, s1] * ...
    # We'll do a big outer product.
    # For M=2, prob_of_combo shape = [K, K]
    # For M=3, shape = [K, K, K], etc.

    # Let's do it for general M with a recursive approach or use np.ix_.
    # We'll do a manual method with broadcasting:

    # Expand each faction's distribution as a multi-dim array that can broadcast
    # e.g. for M=2, we can do outer product: prob_grid = np.outer(strategy_matrix[0], strategy_matrix[1])
    # For M>2, we do repeated expansions.

    prob_grid = None
    for i in range(M):
        dist = strategy_matrix[i]
        if prob_grid is None:
            # shape => (K,)
            prob_grid = dist.reshape([K] + [1] * (M - 1))
        else:
            # multiply with dist in a new axis
            prob_grid = prob_grid * dist.reshape([1] * (i) + [K] + [1] * (M - 1 - i))

    idx_str = (
        ",".join(
            [
                "".join([f"s{j}" for j in range(M)]),
                "i" + "".join([f"s{j}" for j in range(M)]),
            ]
        )
        + "->i"
    )
    return np.einsum(idx_str, prob_grid, payoff_tensor)


# ------------------------------------------------------------------------------
# 1) GAME-THEORETIC FEEDING DECISIONS
# ------------------------------------------------------------------------------
class GameTheoreticFeedingModel:
    """
    Tracks each player's feeding choices over multiple rounds, applying
    game-theoretic logic to predict or enforce alliances, punishments, or
    shifting aggression.

    Potential usage:
      - In single-player, it can model how symbiotes adapt if the player
        consistently starves them vs. occasionally feeding them.
      - In multiplayer, it can identify players who feed often or rarely,
        then symbiotes can show preferential aggression to "greedy" hoarders
        or form ephemeral alliances with consistent feeders.

    Key Methods:
      - record_feeding(player_id, amount): store feeding data
      - get_symbiote_response(player_id): returns an action or aggression
        shift the symbiotes might apply to that player based on feeding patterns
    """

    def __init__(
        self,
        punishment_factor: float = 0.2,
        reward_factor: float = 0.1,
        memory_length: int = 10,
    ):
        """
        Args:
            punishment_factor: How strongly symbiotes punish consistent starvation.
            reward_factor: How strongly symbiotes reward frequent feeders.
            memory_length: How many recent feeding events to consider in calculating
                           strategies. If 0 => entire history.
        """
        self.punishment_factor = punishment_factor
        self.reward_factor = reward_factor
        self.memory_length = memory_length

        # For each player_id, store their feeding amounts in a list
        self.player_feed_history: Dict[str, List[float]] = {}

    def record_feeding(self, player_id: str, feeding_amount: float) -> None:
        """
        Record how many minerals (or how much resource) a particular player fed to
        the symbiotes. This will influence future aggression or alliances.

        Args:
            player_id: Unique identifier for a player/faction.
            feeding_amount: The minerals or resource units fed this turn.
        """
        if player_id not in self.player_feed_history:
            self.player_feed_history[player_id] = []
        self.player_feed_history[player_id].append(feeding_amount)

        # Trim memory
        if self.memory_length > 0:
            while len(self.player_feed_history[player_id]) > self.memory_length:
                self.player_feed_history[player_id].pop(0)

    def compute_average_feeding(self, player_id: str) -> float:
        """
        Returns the average feeding amount for the specified player over
        the tracked memory window.
        """
        history = self.player_feed_history.get(player_id, [])
        return sum(history) / len(history) if history else 0.0

    def get_symbiote_response(self, player_id: str) -> float:
        """
        A basic payoff or "response" from symbiotes to a player's feeding behavior.
        Positive => less aggression or partial alliance.
        Negative => more aggression or punishment.

        The magnitude can be integrated with your aggression system. For instance,
        you might do: new_aggression -= response if response is positive, etc.

        Returns:
            A float in [-punishment_factor, reward_factor], indicating
            the extent of the symbiotes' stance toward that player.
        """
        avg_feed = self.compute_average_feeding(player_id)

        # Example logic: if avg_feed is small => punish,
        # if avg_feed is large => reward. You can calibrate thresholds as needed.
        threshold = 1.0  # if feeding is above 1 unit per turn on average => docile
        if avg_feed < threshold * 0.3:
            # Minimal feeding => punish more severely
            return -self.punishment_factor
        elif avg_feed > threshold:
            # Generous feeding => partial alliance
            return self.reward_factor
        else:
            # In-between => mild punishment or reward scaled proportionally
            ratio = (avg_feed - threshold * 0.3) / (threshold - (threshold * 0.3))
            # ratio in [0..1], map to [-punishment_factor..reward_factor]
            range_val = self.reward_factor + self.punishment_factor
            offset = -self.punishment_factor
            return offset + ratio * range_val


# ------------------------------------------------------------------------------
# 2) VARIABLE MINERAL VALUATION
# ------------------------------------------------------------------------------
class VariableMineralValuation:
    """
    Dynamically adjusts the potency or "value" of each mineral type based on
    usage frequency. If certain minerals are overfed, their effect might drop
    (resistance) or increase (addiction) over time.

    The final "valuation" can feed into your symbiote feeding logic (e.g., how
    much population or mutation is triggered by feeding a certain mineral) or
    your economy (sell price).
    """

    def __init__(
        self,
        mineral_types: List[str],
        initial_values: Dict[str, float],
        usage_sensitivity: float = 0.05,
        adapt_mode: str = "resistance",
    ):
        """
        Args:
            mineral_types: The list of recognized mineral types (e.g. ["common","rare","precious","anomaly"]).
            initial_values: A dict mapping each mineral type to an initial valuation or potency.
            usage_sensitivity: How strongly the valuation changes per usage. A higher number => bigger adaptation.
            adapt_mode: "resistance" => overuse leads to reduced value,
                        "addiction" => overuse leads to increased value,
                        or "balanced" => a dynamic push-pull approach you can define.
        """
        self.mineral_types = mineral_types
        self.values = initial_values.copy()
        self.usage_sensitivity = usage_sensitivity
        self.adapt_mode = adapt_mode

        # Track usage counts
        self.usage_counts: Dict[str, float] = {m: 0.0 for m in mineral_types}

    def record_usage(self, mineral_type: str, amount: float) -> None:
        """
        Register that 'amount' of a particular mineral has been used (fed, etc.).
        The usage_count will eventually drive changes in valuation.
        """
        if mineral_type not in self.usage_counts:
            return
        self.usage_counts[mineral_type] += amount

    def update_valuations(self) -> None:
        """
        Adapts each mineral's valuation based on usage_counts. The approach
        depends on adapt_mode:
          - "resistance": overuse => reduce value
          - "addiction": overuse => increase value
          - "balanced": e.g., if usage above a threshold => reduce, else => increase
        Then resets usage_counts so each call addresses usage in that last period.
        """
        for mtype in self.mineral_types:
            old_val = self.values.get(mtype, 1.0)
            usage = self.usage_counts.get(mtype, 0.0)
            if usage <= 0:
                continue

            if self.adapt_mode == "resistance":
                # Overuse => reduce
                new_val = max(0.1, old_val - usage * self.usage_sensitivity)
                self.values[mtype] = new_val
            elif self.adapt_mode == "addiction":
                # Overuse => raise
                new_val = old_val + usage * self.usage_sensitivity
                # Potentially cap if you want
                self.values[mtype] = min(100.0, new_val)
            elif self.adapt_mode == "balanced":
                # If usage above certain threshold => reduce, else => increase
                threshold = 10.0
                if usage > threshold:
                    ratio = (usage - threshold) / threshold
                    new_val = max(0.1, old_val * (1.0 - ratio * 0.2))
                    self.values[mtype] = new_val
                else:
                    # Light usage => slight increase
                    inc_factor = 1.0 + (usage / threshold * 0.1)
                    self.values[mtype] = old_val * inc_factor

        # Reset usage counts after adaptation
        for mtype in self.mineral_types:
            self.usage_counts[mtype] = 0.0

    def get_valuation(self, mineral_type: str) -> float:
        """
        Returns the current valuation/potency for a specified mineral type.
        """
        return self.values.get(mineral_type, 1.0)


# ------------------------------------------------------------------------------
# 3) ADAPTIVE AGGRESSION COST-BENEFIT: SymbioteAttackEVCalculator
# ------------------------------------------------------------------------------
class SymbioteAttackEVCalculator:
    """
    Computes the expected value of an attack vs. waiting/growing, factoring in:
      - Probability of victory (p_victory)
      - Probability of defeat (p_defeat = 1 - p_victory)
      - spoils from victory
      - losses from defeat
      - potential future payoffs from waiting

    The user can integrate this into an RL or evolutionary algorithm that tries
    to choose the action (attack/wait/grow) with the highest EV.
    """

    def __init__(self, discount_factor: float = 0.9):
        """
        Args:
            discount_factor: If the symbiotes value future payoffs less than immediate payoffs,
                            they discount waiting’s reward. E.g., 0.9 => future is 90% as valuable.
        """
        self.discount_factor = discount_factor

    def calculate_ev_attack(
        self, p_victory: float, spoils_value: float, p_defeat: float, defeat_loss: float
    ) -> float:
        """
        EV(attack) = p_victory * spoils_value - p_defeat * defeat_loss

        Args:
            p_victory: Probability of winning the engagement (0..1).
            spoils_value: Gains from a successful attack (minerals, territory, kills).
            p_defeat: Probability of losing (should be 1 - p_victory, but you can pass it explicitly).
            defeat_loss: Penalty from losing (lost symbiotes, territory, etc.).

        Returns:
            The expected value of attacking.
        """
        return (p_victory * spoils_value) - (p_defeat * defeat_loss)

    def calculate_ev_wait(
        self, immediate_gain: float, future_gain_estimate: float
    ) -> float:
        """
        EV(wait) = immediate_gain + discount_factor * future_gain_estimate

        Where immediate_gain might be, for example, the resources gained by
        not attacking right now (like continuing to gather minerals or feed),
        and future_gain_estimate is how much we expect to gain from an attack
        (or another action) in the near future, discounted by self.discount_factor.

        Args:
            immediate_gain: How much is gained by not attacking in this turn (e.g. more
                            population growth, fewer casualties).
            future_gain_estimate: The potential value if a future attack is more successful
                                  (or if continuing to wait yields synergy with expansions).

        Returns:
            The expected value of waiting/growing for one time step.
        """
        return immediate_gain + self.discount_factor * future_gain_estimate

    def pick_best_action(
        self,
        ev_attack: float,
        ev_wait: float,
        optional_actions: Dict[str, float] = None,
    ) -> str:
        """
        Compares EV(attack), EV(wait), and any optional actions (like "harass" with
        known EV) to pick the best action. Returns a string naming that action.

        Args:
            ev_attack: expected value of a direct attack now
            ev_wait: expected value of waiting or growing
            optional_actions: a dict of {action_name: EV_value}, e.g. {"harass": 2.0}

        Returns:
            A string indicating which action yields the highest EV overall.
        """
        candidates = {"attack": ev_attack, "wait": ev_wait}
        if optional_actions:
            candidates |= optional_actions

        best_action = None
        best_ev = float("-inf")
        for action_name, ev_val in candidates.items():
            if ev_val > best_ev:
                best_ev = ev_val
                best_action = action_name

        return best_action
