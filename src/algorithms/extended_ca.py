"""
extended_ca.py

Advanced cellular automaton (CA) techniques, building on a standard
Game of Life–style or other CA to handle custom neighborhoods, multiple
layers, rule evolution, and multi-scale (zoom-level) simulations. These
classes provide the tools to create richer, more dynamic CA-driven
symbiote ecosystems.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------
1. CustomNeighborhoodAutomaton
   - Allows definition of arbitrary neighbor positions or extended radii
     for computing births and survivals.
   - Can incorporate directionality or weighting to certain neighbors.

2. LayeredAutomaton
   - Manages multiple data layers (e.g., symbiote biomass, resource levels,
     temperature) that factor into birth/survival rules. 
   - Each layer updates in sync but can influence each other, allowing complex
     interactions beyond a single CA grid.

3. EvolvingCARules
   - Introduces the concept of a “rule genome,” enabling mutation of birth
     and survival sets over time. Different colonies or regions can diverge
     into distinct rule sets (e.g., Seeds, HighLife, custom rules).
   - Tracks rule sets in a dictionary or array per region, possibly applying
     different CA logic in each cluster.

4. MultiScaleAutomaton
   - Runs CA at multiple scales (fine grid vs. coarse grid). Large-scale
     expansions might be handled at a coarse resolution, while local details
     or combat happen at a finer resolution.
   - Periodically synchronizes or merges data from the different scales.

No usage examples are included per instructions. Integrate or subclass these
classes in your main symbiote evolution code, adjusting parameters for your
game’s needs.

--------------------------------------------------------------------------------
LICENSE / COPYRIGHT NOTICE
--------------------------------------------------------------------------------
Copyright (c) 2025 ...
All rights reserved.
"""


import itertools
import numpy as np
import random
from typing import Set, Tuple, Dict, List


# ------------------------------------------------------------------------------
# 1) CUSTOM NEIGHBORHOODS
# ------------------------------------------------------------------------------
class CustomNeighborhoodAutomaton:
    """
    A cellular automaton that uses an arbitrary set of neighbor offsets
    (rather than the standard Moore or von Neumann neighborhoods). This
    can simulate tentacle growth, extended hives, or directional expansions.
    """

    def __init__(
        self,
        width: int,
        height: int,
        birth_set: Set[int],
        survival_set: Set[int],
        neighbor_offsets: List[Tuple[int, int]],
    ):
        """
        Args:
            width, height: Dimensions of the CA grid.
            birth_set: Number of neighbors that cause a dead cell to become alive.
            survival_set: Number of neighbors that keep a live cell alive.
            neighbor_offsets: A list of (dx, dy) relative positions that define
                              the neighborhood. For example, for an 8-neighbor
                              Moore: [(-1,-1),(-1,0),...].
        """
        self.width = width
        self.height = height
        self.birth_set = birth_set
        self.survival_set = survival_set
        self.neighbor_offsets = neighbor_offsets

        # 2D grid: 1 = alive, 0 = dead
        self.grid = np.zeros((width, height), dtype=np.int8)

    def step(self) -> None:
        """
        Advances the CA by one generation using the custom neighborhood
        definition. Wrap-around or other boundary conditions can be applied
        as needed (by default, we do wrap for demonstration).
        """
        new_grid = np.zeros_like(self.grid)
        for x in range(self.width):
            for y in range(self.height):
                alive_neighbors = 0
                for dx, dy in self.neighbor_offsets:
                    nx = (x + dx) % self.width
                    ny = (y + dy) % self.height
                    alive_neighbors += self.grid[nx, ny]

                cell_state = self.grid[x, y]
                if (
                    cell_state == 1
                    and alive_neighbors in self.survival_set
                    or cell_state != 1
                    and alive_neighbors in self.birth_set
                ):
                    new_grid[x, y] = 1
        self.grid = new_grid

    def set_cell(self, x: int, y: int, state: int) -> None:
        """
        Manually set a cell’s state.
        """
        self.grid[x % self.width, y % self.height] = state

    def get_cell(self, x: int, y: int) -> int:
        """
        Returns the state of a cell, wrapping if out of bounds.
        """
        return self.grid[x % self.width, y % self.height]


# ------------------------------------------------------------------------------
# 2) MULTIPLE LAYERS / RESOURCE LAYERS
# ------------------------------------------------------------------------------
class LayeredAutomaton:
    """
    Maintains multiple layers, each of which can store some numeric data
    (e.g. symbiote presence, resource levels, temperature). You can define
    separate birth/survival logic that draws on any combination of these layers.

    The CA updates happen in steps, possibly referencing multiple layers for
    a single cell’s next state. One layer might be “symbiote biomass,” another
    might be “nutrient resources,” etc.
    """

    def __init__(
        self,
        width: int,
        height: int,
        layer_names: List[str],
        update_rules: Dict[str, callable],
    ):
        """
        Args:
            width, height: grid dimensions.
            layer_names: e.g. ["biomass","resource","temperature"].
            update_rules: dict mapping layer_name -> update function. The function
                          is called each step for that layer, signature:
                             func(layers, x, y) -> new_value
                          where layers is a dict of {layer_name: 2D array}.
        """
        self.width = width
        self.height = height
        self.layer_names = layer_names
        self.update_rules = update_rules

        # For each layer, store a 2D numpy array
        self.layers: Dict[str, np.ndarray] = {}
        for ln in layer_names:
            self.layers[ln] = np.zeros((width, height), dtype=np.float32)

    def step(self) -> None:
        """
        Updates each layer by calling its update function across the grid.
        You may need to store temporary arrays to avoid partial overwrites.
        """
        new_layers = {ln: np.zeros_like(self.layers[ln]) for ln in self.layer_names}
        # For each cell, call each layer’s update function
        for x in range(self.width):
            for y in range(self.height):
                # Provide the entire set of layers to the rule
                for ln in self.layer_names:
                    rule_func = self.update_rules.get(ln, None)
                    if rule_func is None:
                        new_layers[ln][x, y] = self.layers[ln][x, y]
                    else:
                        new_val = rule_func(self.layers, x, y)
                        new_layers[ln][x, y] = new_val

        self.layers = new_layers

    def set_value(self, layer_name: str, x: int, y: int, value: float) -> None:
        """
        Set a cell’s value in a specific layer.
        """
        self.layers[layer_name][x % self.width, y % self.height] = value

    def get_value(self, layer_name: str, x: int, y: int) -> float:
        """
        Retrieve the cell’s value in a specific layer.
        """
        return self.layers[layer_name][x % self.width, y % self.height]


# ------------------------------------------------------------------------------
# 3) AUTOMATON RULE EVOLUTION
# ------------------------------------------------------------------------------
class EvolvingCARules:
    """
    Manages multiple CA rule sets across a grid, letting them mutate over time.
    Each cell (or region) can have its own 'rule genome' specifying which birth
    and survival sets it uses. Over many steps, these rule sets can diverge or
    converge, leading to a patchwork of CA behaviors.
    """

    def __init__(
        self,
        width: int,
        height: int,
        initial_rule: Tuple[Set[int], Set[int]],
        mutation_prob: float = 0.01,
    ):
        """
        Args:
            width, height: Grid dimensions.
            initial_rule: A (birth_set, survival_set) for initialization.
                          e.g. ( {3}, {2,3} ) for Conway’s Game of Life.
            mutation_prob: Probability of a rule mutation event each step for
                           a given cell or region.
        """
        self.width = width
        self.height = height
        self.mutation_prob = mutation_prob

        # Each cell has a (birth_set, survival_set) genome
        # We'll store them as strings or tuples in 2D arrays for convenience
        self.birth_rules = np.empty((width, height), dtype=object)
        self.survival_rules = np.empty((width, height), dtype=object)

        # For the CA state (alive/dead), we keep a separate grid
        self.grid = np.zeros((width, height), dtype=np.int8)

        # Initialize all cells with the same rule
        birth_init, surv_init = initial_rule
        for x, y in itertools.product(range(width), range(height)):
            self.birth_rules[x, y] = set(birth_init)
            self.survival_rules[x, y] = set(surv_init)

    def step(self) -> None:
        """
        Perform one step of the multi-rule CA:
         1) Possibly mutate the rule sets in some cells.
         2) Evolve each cell using its local (birth,survival) sets.
        """
        new_grid = np.zeros_like(self.grid)

        # 1) Apply rule mutations
        self._apply_rule_mutations()
        
        # 2) Apply local rules to generate the new grid
        self._apply_local_rules(new_grid)

        self.grid = new_grid
        
    def _apply_rule_mutations(self) -> None:
        """
        Apply potential rule mutations to each cell based on mutation probability.
        """
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < self.mutation_prob:
                    self._mutate_cell_rule(x, y)
                    
    def _apply_local_rules(self, new_grid: np.ndarray) -> None:
        """
        Apply CA rules to each cell to determine its next state.
        
        Args:
            new_grid: The grid to store the next generation states
        """
        for x in range(self.width):
            for y in range(self.height):
                self._update_cell_state(x, y, new_grid)
                
    def _update_cell_state(self, x: int, y: int, new_grid: np.ndarray) -> None:
        """
        Update a single cell's state based on local rules.
        
        Args:
            x: X coordinate of cell
            y: Y coordinate of cell
            new_grid: Grid to store the updated state
        """
        neighbors = self._count_neighbors(x, y)
        cell_alive = self.grid[x, y] == 1
        
        # Apply birth or survival rules based on current state
        rule_set = self.survival_rules[x, y] if cell_alive else self.birth_rules[x, y]
        if neighbors in rule_set:
            new_grid[x, y] = 1

    def _count_neighbors(self, x: int, y: int) -> int:
        """
        Counts 8-neighborhood for simplicity.
        You could expand to a custom neighborhood approach if you prefer.
        """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                count += self.grid[nx, ny]
        return count

    def _mutate_cell_rule(self, x: int, y: int) -> None:
        """
        Introduces a random mutation to the rule sets for cell (x,y).
        e.g., randomly add or remove a neighbor count in birth or survival sets.
        """
        # Weighted chance to mutate birth vs survival
        target_set = self.birth_rules[x, y] if random.random() < 0.5 else self.survival_rules[x, y]
        
        # Either remove or add an element
        if random.random() < 0.5 and len(target_set) > 0:
            # Remove a random element
            target_set.remove(random.choice(list(target_set)))
        else:
            # Add a random neighbor count
            new_num = random.randint(0, 8)
            target_set.add(new_num)

    def set_cell(self, x: int, y: int, state: int) -> None:
        self.grid[x % self.width, y % self.height] = state

    def get_cell(self, x: int, y: int) -> int:
        return self.grid[x % self.width, y % self.height]


# ------------------------------------------------------------------------------
# 4) MULTISCALE CA
# ------------------------------------------------------------------------------
class MultiScaleAutomaton:
    """
    Runs multiple CA layers at different resolutions. For example:
      - A coarse layer for large-scale territory expansions,
      - A finer layer for local detail or advanced combat logic.

    Periodically merges or syncs the layers, ensuring consistency. This can
    reduce computation while still allowing high-detail behaviors in focal areas.
    """

    def __init__(
        self,
        fine_width: int,
        fine_height: int,
        coarse_width: int,
        coarse_height: int,
        fine_rule: Tuple[Set[int], Set[int]],
        coarse_rule: Tuple[Set[int], Set[int]],
    ):
        """
        Args:
            fine_width, fine_height: Dimensions for the high-resolution grid.
            coarse_width, coarse_height: Dimensions for the coarse grid.
            fine_rule: (birth_set, survival_set) for the fine-scale CA.
            coarse_rule: (birth_set, survival_set) for the coarse-scale CA.
        """
        self.fine_width = fine_width
        self.fine_height = fine_height
        self.coarse_width = coarse_width
        self.coarse_height = coarse_height

        # For simplicity, we store two standard CA grids
        b_fine, s_fine = fine_rule
        self.fine_grid = np.zeros((fine_width, fine_height), dtype=np.int8)

        b_coarse, s_coarse = coarse_rule
        self.coarse_grid = np.zeros((coarse_width, coarse_height), dtype=np.int8)

        self.birth_fine = set(b_fine)
        self.survival_fine = set(s_fine)
        self.birth_coarse = set(b_coarse)
        self.survival_coarse = set(s_coarse)

    def step_fine(self, steps: int = 1) -> None:
        """
        Runs the fine-scale CA for a specified number of steps.
        """
        for _ in range(steps):
            self.fine_grid = self._ca_step(
                self.fine_grid,
                self.fine_width,
                self.fine_height,
                self.birth_fine,
                self.survival_fine,
            )

    def step_coarse(self, steps: int = 1) -> None:
        """
        Runs the coarse-scale CA for a specified number of steps.
        """
        for _ in range(steps):
            self.coarse_grid = self._ca_step(
                self.coarse_grid,
                self.coarse_width,
                self.coarse_height,
                self.birth_coarse,
                self.survival_coarse,
            )

    def sync_coarse_to_fine(self) -> None:
        """
        Merge or sync the coarse grid into the fine grid.
        For example, you might replicate a coarse cell's state across the
        corresponding region in the fine grid.
        """
        x_ratio = self.fine_width // self.coarse_width
        y_ratio = self.fine_height // self.coarse_height

        for cx, cy in itertools.product(range(self.coarse_width), range(self.coarse_height)):
            cval = self.coarse_grid[cx, cy]
            # Write cval to the corresponding block in fine_grid
            start_x = cx * x_ratio
            start_y = cy * y_ratio
            for fx, fy in itertools.product(range(start_x, start_x + x_ratio), range(start_y, start_y + y_ratio)):
                self.fine_grid[fx, fy] = cval

    def sync_fine_to_coarse(self) -> None:
        """
        Optionally, gather an aggregate from the fine grid to update the coarse grid.
        For instance, if more than half of the fine cells in a block are alive,
        set the coarse cell to 1, otherwise 0.
        """
        x_ratio = self.fine_width // self.coarse_width
        y_ratio = self.fine_height // self.coarse_height

        # Process each coarse grid cell
        for cx in range(self.coarse_width):
            for cy in range(self.coarse_height):
                self._update_coarse_cell_from_fine_block(cx, cy, x_ratio, y_ratio)
                
    def _update_coarse_cell_from_fine_block(self, cx: int, cy: int, x_ratio: int, y_ratio: int) -> None:
        """
        Update a single coarse cell based on its corresponding fine grid block.
        
        Args:
            cx: X coordinate in coarse grid
            cy: Y coordinate in coarse grid
            x_ratio: X scale ratio between fine and coarse grids
            y_ratio: Y scale ratio between fine and coarse grids
        """
        # Calculate the boundaries of the fine grid block
        start_x = cx * x_ratio
        start_y = cy * y_ratio
        
        # Extract the corresponding block from the fine grid
        block = self.fine_grid[
            start_x : start_x + x_ratio, start_y : start_y + y_ratio
        ]
        
        # Calculate alive cells and threshold
        alive_count = np.sum(block)
        total_cells = x_ratio * y_ratio
        
        # Set coarse cell state based on majority rule
        self.coarse_grid[cx, cy] = 1 if alive_count > (total_cells // 2) else 0

    def _ca_step(
        self,
        grid: np.ndarray,
        width: int,
        height: int,
        birth_set: Set[int],
        survival_set: Set[int],
    ) -> np.ndarray:
        """
        Single CA step for a standard 8-neighbor Moore neighborhood, with wrapping.
        This is reused for both fine_grid and coarse_grid updates.
        """
        new_grid = np.zeros_like(grid)
        for x, y in itertools.product(range(width), range(height)):
            neigh = self._count_cell_neighbors(grid, x, y, width, height)
            new_grid[x, y] = self._determine_new_state(grid[x, y], neigh, birth_set, survival_set)
        return new_grid
        
    def _count_cell_neighbors(
        self, grid: np.ndarray, x: int, y: int, width: int, height: int
    ) -> int:
        """
        Count the number of living neighbors for a cell in the grid.
        
        Args:
            grid: The cellular automaton grid
            x: X coordinate of cell
            y: Y coordinate of cell
            width: Width of the grid
            height: Height of the grid
            
        Returns:
            Count of living neighbors
        """
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % width
                ny = (y + dy) % height
                count += grid[nx, ny]
        return count
        
    def _determine_new_state(
        self, current_state: int, neighbors: int, birth_set: Set[int], survival_set: Set[int]
    ) -> int:
        """
        Determine the new state of a cell based on its current state and neighbors.
        
        Args:
            current_state: Current state of the cell (0 or 1)
            neighbors: Number of living neighbors
            birth_set: Set of neighbor counts that cause birth
            survival_set: Set of neighbor counts that allow survival
            
        Returns:
            New state of the cell (0 or 1)
        """
        if current_state == 1 and neighbors in survival_set:
            return 1
        return 1 if current_state == 0 and neighbors in birth_set else 0
