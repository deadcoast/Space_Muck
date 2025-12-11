"""
src/algorithms/pattern_analysis.py

Provides advanced pattern detection and analysis utilities for cellular automata
and colony-based grids. Includes detection of complex oscillators (pulsars, toads,
gliders), Voronoi partitioning of labeled regions for territory assignment, and
colony merging/assimilation mechanics. Each feature can feed into AI logic to help
the symbiotes decide strategies based on established patterns or territory overlaps.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------
1. AdvancedPatternAnalyzer
   - Identify multi-step oscillators such as pulsars, toads, or glider cycles.
   - Maintains a recent history of grid states to detect repeating or shifting patterns.
   - Helps AI gauge the “maturity” of a colony based on stable or oscillatory structures.

2. VoronoiTerritoryPartitioner
   - After labeling colonies (connected components), compute approximate Voronoi
     regions for each colonys center or key point.
   - Useful for deciding how territories are partitioned among multiple colonies,
     or how symbiotes decide who to fight or merge with.

3. ColonyMerger
   - Builds on a labeled colony grid, adding a mechanism to merge weaker colonies
     into stronger ones (or partially assimilate them).
   - Can combine or average “genomes” if you track attributes (e.g., aggression base,
     mutation rate) for each colony.

No usage examples included (per instructions). Integrate these classes/functions
into your main CA or symbiote simulation code.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025 ...
All rights reserved.
"""

from collections import deque
from typing import Dict, Tuple

import numpy as np


# ------------------------------------------------------------------------------
# 1) ADVANCED PATTERN RECOGNITION
# ------------------------------------------------------------------------------
class AdvancedPatternAnalyzer:
    """
    Detects complex patterns and oscillators in a cellular automaton grid. For
    instance, it can store a short history of grids and look for repeats to
    identify multi-step oscillators like toads (period 2), pulsars (period 3),
    or glider-like translations. The user can adapt the detection heuristics
    to match the game’s CA rules if they differ from standard Life.

    Common workflow:
      - Each timestep, call add_state(current_grid).
      - Periodically call detect_oscillator() or detect_glider() to see if a known
        repeating or moving pattern is recognized.
    """

    def __init__(self, max_period: int = 15, max_history: int = 50):
        """
        Args:
            max_period: The maximum oscillator period we check for repeating states.
            max_history: How many recent states to store. If states exceed this,
                         older ones are discarded. Larger history = more memory usage.
        """
        self.max_period = max_period
        self.max_history = max_history
        self.history = deque()  # store (grid_state, time_idx) or just grid states
        self.time_index = 0

    def add_state(self, grid: np.ndarray) -> None:
        """
        Appends the current grid snapshot to history for pattern analysis.
        The grid is copied to avoid referencing external changes.
        """
        if len(self.history) >= self.max_history:
            self.history.popleft()
        self.history.append((grid.copy(), self.time_index))
        self.time_index += 1

    def detect_oscillator(self) -> int:
        """
        Checks if the most recent state matches any earlier state within
        max_period steps. Returns the period of the discovered oscillator,
        or 0 if none found.

        This approach looks for exact grid repeats. For multi-step or partial
        repeats (like a shift for gliders), see detect_glider().
        """
        if len(self.history) < 2:
            return 0

        recent_grid, recent_time = self.history[-1]
        # We only need to check up to max_period states back
        limit = min(len(self.history) - 1, self.max_period)
        for i in range(1, limit + 1):
            compare_grid, compare_time = self.history[-1 - i]
            if np.array_equal(recent_grid, compare_grid):
                # Found a repeat => period = recent_time - compare_time
                return recent_time - compare_time
        return 0

    @staticmethod
    def detect_glider() -> int:
        """
        A placeholder for more advanced detection of moving patterns like gliders.
        Typically requires analyzing translations across multiple timesteps, or
        pattern matching. Returns the period if a known glider is detected, else 0.

        Expand this method to handle searching for known moving patterns in
        standard CA rule sets (like Conway's Life). If the game rules differ,
        you'd need specialized pattern definitions.
        """
        # For demonstration, we just return 0 (no detection).
        # Real implementation might store consecutive states, attempt to align them
        # via x/y shifts to see if a pattern reappears.
        return 0

    def detect_stable_structures(self, threshold: int = 3) -> bool:
        """
        Determines if the CA has converged to a stable or near-stable configuration
        by comparing the last X states. If they are identical or very similar,
        we consider the pattern stable.

        Args:
            threshold: how many recent states to compare for stability.

        Returns:
            True if the last 'threshold' states are identical or nearly so, else False.
        """
        if len(self.history) < threshold:
            return False

        # Compare the last 'threshold' grids to see if they match
        ref_grid, _ = self.history[-1]
        for i in range(2, threshold + 1):
            grid_i, _ = self.history[-i]
            if not np.array_equal(ref_grid, grid_i):
                return False
        return True


# ------------------------------------------------------------------------------
# 2) VORONOI / REGION PARTITIONING
# ------------------------------------------------------------------------------
class VoronoiTerritoryPartitioner:
    """
    Given a labeled colony map (e.g., from connected component labeling),
    this class approximates a Voronoi partition for each colony’s “center.”
    A center can be the colony centroid or a chosen reference point.

    Usage:
      1) Label the grid, find each colony’s center (like center of mass).
      2) Initialize VoronoiTerritoryPartitioner with these points.
      3) call compute_voronoi() to get a partition where each cell is assigned
         to the nearest colony center.
    """

    def __init__(
        self, width: int, height: int, centers: Dict[int, Tuple[float, float]]
    ):
        """
        Args:
            width, height: Dimensions of the grid.
            centers: A dict mapping colony_id -> (center_x, center_y). Each
                     center is a float or int coordinate. We use Euclidean
                     distance for partitioning.
        """
        self.width = width
        self.height = height
        self.centers = centers  # e.g. { 1:(10.3, 20.9), 2:(30.1, 5.7), ... }

    def compute_voronoi(self) -> np.ndarray:
        """
        Returns a 2D numpy array (width x height) where each cell is assigned
        the ID of the nearest colony center. If no centers exist, returns an
        array of -1’s.

        The distance measure is standard Euclidean. No advanced weighting or
        boundary wrap is done. You can adapt as needed.
        """
        if not self.centers:
            return -1 * np.ones((self.width, self.height), dtype=np.int32)

        # Convert centers to list for iteration
        center_items = list(self.centers.items())  # [(colony_id, (cx,cy)),...]

        partition_map = np.zeros((self.width, self.height), dtype=np.int32)

        for x in range(self.width):
            for y in range(self.height):
                best_id = -1
                best_dist = float("inf")
                for colony_id, (cx, cy) in center_items:
                    dx = x - cx
                    dy = y - cy
                    dist2 = dx * dx + dy * dy
                    if dist2 < best_dist:
                        best_dist = dist2
                        best_id = colony_id
                partition_map[x, y] = best_id

        return partition_map


# ------------------------------------------------------------------------------
# 3) COLONY MERGER / CONVERSION
# ------------------------------------------------------------------------------
class ColonyMerger:
    """
    Implements logic for merging or assimilating weaker colonies after
    interactions (combat, bridging, or alliances). Designed to integrate with
    a labeled grid or a data structure representing each colony’s “genome” or
    stats (e.g., aggression_base, mutation_rate, etc.).
    """

    def __init__(self):
        """
        You can store references to a global colony dictionary or
        pass it in as needed for merges.
        """
        pass

    def merge_colonies(
        self,
        target_id: int,
        source_id: int,
        labeled_grid: np.ndarray,
        colony_data: Dict[int, Dict[str, float]],
        assimilation_ratio: float = 0.5,
    ) -> None:
        """
        Assimilates the 'source_id' colony into 'target_id'. The labeled_grid
        is updated so that cells with 'source_id' become 'target_id'. The
        'colony_data' is updated by merging the two sets of traits.

        Args:
            target_id: ID of the winner or absorbing colony.
            source_id: ID of the loser or smaller colony to be merged.
            labeled_grid: A 2D array where each cell is assigned an integer colony ID.
            colony_data: A dict mapping colony_id -> {trait_name: trait_value}, storing
                         relevant attributes or “genome” info for each colony.
            assimilation_ratio: 0 < ratio < 1.
                E.g., 0.5 => average the traits. 0.7 => 70% from target colony,
                30% from source, etc.
        """
        # Update the grid cells to target_id
        labeled_grid[labeled_grid == source_id] = target_id

        # Merge genome data if both colonies exist in colony_data
        if source_id in colony_data and target_id in colony_data:
            self._merge_genomes(
                colony_data[target_id], colony_data[source_id], assimilation_ratio
            )
            # Remove source_id from colony_data
            del colony_data[source_id]

    @staticmethod
    def _merge_genomes(
        target_genome: Dict[str, float],
        source_genome: Dict[str, float],
        alpha: float,
    ) -> None:
        """
        Merge or average the source genome into the target genome. The user may also
        define a more complex logic than a simple weighted average if the game demands
        it (like keeping the higher aggression, etc.).
        """
        for k, src_val in source_genome.items():
            tgt_val = target_genome.get(k, 0.0)
            # Weighted combination
            new_val = (alpha * tgt_val) + ((1.0 - alpha) * src_val)
            target_genome[k] = new_val
