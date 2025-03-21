"""
src/algorithms/spatial_economic_model.py

Combines PDE-based spatial dynamics with economic decision-making, creating
a spatially-aware resource optimization system. This module builds upon:
1) economy_decision.py - For economic and decision modeling
2) pde_spatial.py - For spatial dynamics and resource distribution

This integration allows for:
- Resource distribution based on spatial dynamics and concentration gradients
- Area-dependent constraints in economic optimization
- Spatially-aware payoff calculations for multi-faction decision making
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

from src.algorithms.economy_decision import (
    MineralResourceOptimizer,
    compute_multi_faction_payoff,
)
from src.algorithms.pde_spatial import AdvectionDiffusionSystem


class SpatialMineralOptimizer(MineralResourceOptimizer):
    """
    Extends the MineralResourceOptimizer to incorporate PDE-based spatial constraints.

    This allows resource optimization to account for:
    1) Spatial distribution of resources based on PDE models
    2) Location-dependent constraints on feeding and selling
    3) Gradient-based resource value adjustments
    """

    def __init__(
        self,
        mineral_types: List[str],
        grid_shape: Tuple[int, int],
        diffusion_coeffs: Dict[str, float],
        dt: float = 1.0,
        boundary_mode: str = "wrap",
    ):
        """
        Initialize with both mineral economic data and spatial PDE parameters.

        Args:
            mineral_types: List of mineral type identifiers (e.g. ["rare", "common"])
            grid_shape: (width, height) of the spatial grid
            diffusion_coeffs: Diffusion coefficients for each mineral type
            dt: Time step for PDE updates
            boundary_mode: Boundary handling for PDE ("wrap", "reflect", "constant")
        """
        super().__init__(mineral_types)

        # Create a PDE system to track mineral distributions
        self.pde_system = AdvectionDiffusionSystem(
            grid_shape=grid_shape,
            diffusion_coeffs={
                m_type: diffusion_coeffs.get(m_type, 0.1) for m_type in mineral_types
            },
            dt=dt,
            boundary_mode=boundary_mode,
        )

        # Initialize spatial fields for each mineral type
        for m_type in mineral_types:
            self.pde_system.set_field(m_type, 0.0)

        # Define grid dimensions for convenience
        self.width, self.height = grid_shape

    def update_mineral_distribution(self, resource_map: Dict[str, np.ndarray]) -> None:
        """
        Update the spatial distribution of minerals based on external resource map.

        Args:
            resource_map: Dictionary mapping mineral_type to 2D concentration arrays
        """
        for m_type, concentration in resource_map.items():
            if m_type in self.mineral_types:
                self.pde_system.set_field_array(m_type, concentration)

    def add_mineral_source(
        self, mineral_type: str, position: Tuple[int, int], radius: int, amount: float
    ) -> None:
        """
        Add a localized source of minerals at a specific position.

        Args:
            mineral_type: Type of mineral to add
            position: (x, y) coordinates for the source
            radius: Radius of effect
            amount: Amount to add
        """
        if mineral_type not in self.mineral_types:
            raise ValueError(f"Unknown mineral type: {mineral_type}")

        # Get the current field
        field = self.pde_system.fields[mineral_type].copy()

        # Create a circular mask for the source
        x, y = np.ogrid[: self.width, : self.height]
        cx, cy = position
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist_from_center <= radius

        # Add the mineral to the field
        field[mask] += amount

        # Update the field in the PDE system
        self.pde_system.set_field_array(mineral_type, field)

    def step_diffusion(self, steps: int = 1) -> None:
        """
        Evolve the PDE system for the specified number of steps.

        Args:
            steps: Number of PDE updates to perform
        """
        for _ in range(steps):
            self.pde_system.step()

    def get_location_specific_feeding_plan(
        self,
        position: Tuple[int, int],
        radius: int,
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None,
        aggression_coef: float = 0.0,
    ) -> Dict[str, float]:
        """
        Create a feeding plan based on minerals available in a specific area.

        Args:
            position: (x, y) center of the area to consider
            radius: Radius to consider around the center
            feed_benefits: Benefits gained from feeding each mineral
            sell_values: Market values for selling each mineral
            min_feeding_requirements: Minimum amounts to feed
            aggression_coef: Balance between profit and appeasement

        Returns:
            Dictionary with optimized feeding plan
        """
        # Create a circular mask for the area
        x, y = np.ogrid[: self.width, : self.height]
        cx, cy = position
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist_from_center <= radius

        # Calculate total minerals available in the area
        total_minerals = {}
        for m_type in self.mineral_types:
            field = self.pde_system.fields[m_type]
            total_minerals[m_type] = float(np.sum(field[mask]))

        # Use the base class linear optimization with spatially-aware mineral quantities
        return self.linear_optimize_feeding(
            total_minerals,
            feed_benefits,
            sell_values,
            min_feeding_requirements,
            aggression_coef,
        )

    def compute_spatial_payoff_matrix(
        self,
        faction_positions: List[Tuple[int, int]],
        strategy_matrix: np.ndarray,
        base_payoff_tensor: np.ndarray,
        influence_radius: int = 5,
    ) -> np.ndarray:
        """
        Calculate a spatially-aware payoff matrix that accounts for resource distribution.

        Args:
            faction_positions: List of (x,y) positions for each faction
            strategy_matrix: Strategy distribution for each faction [n_factions, n_strategies]
            base_payoff_tensor: Base payoff tensor [n_factions, n_strat1, n_strat2, ...]
            influence_radius: Radius to consider for resource influence

        Returns:
            Modified payoff tensor accounting for spatial mineral distribution
        """
        n_factions = len(faction_positions)

        # Create a copy of the base payoff tensor to modify
        spatial_payoff = base_payoff_tensor.copy()

        # Resource optimization for allocating minerals efficiently across factions
        # This can be formulated as a linear programming problem
        if n_factions > 1:
            self._optimize_resource_allocation(
                n_factions, faction_positions, influence_radius
            )

        # For each faction, compute a spatial modifier based on local resources
        for i, position in enumerate(faction_positions):
            # Get local resource abundance for this faction
            local_resources = {}
            for m_type in self.mineral_types:
                field = self.pde_system.fields[m_type]

                # Create circular mask
                x, y = np.ogrid[: self.width, : self.height]
                cx, cy = position
                dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                mask = dist_from_center <= influence_radius

                # Calculate average concentration in the area
                local_resources[m_type] = float(np.mean(field[mask]))

            # Calculate a resource multiplier (example: average of normalized resources)
            # This is just one possible approach - could be more sophisticated
            resource_values = np.array(
                [local_resources.get(m, 0.0) for m in self.mineral_types]
            )
            if resource_values.sum() > 0:
                resource_multiplier = (
                    1.0 + 0.5 * (resource_values / resource_values.max()).mean()
                )
            else:
                resource_multiplier = 1.0

            # Apply the multiplier to all payoffs for this faction
            # This modifies the faction's slice of the payoff tensor
            idx = [slice(None)] * (base_payoff_tensor.ndim)
            idx[0] = i  # Set the faction index
            spatial_payoff[tuple(idx)] *= resource_multiplier

        # Calculate the final payoffs using the standard compute function
        return compute_multi_faction_payoff(strategy_matrix, spatial_payoff)

    def _optimize_resource_allocation(
        self,
        n_factions: int,
        faction_positions: List[Tuple[int, int]],
        influence_radius: int,
    ):
        """
        Optimize the allocation of minerals across factions using linear programming.

        Args:
            n_factions: Number of factions to allocate resources among
            faction_positions: List of (x,y) positions for each faction
            influence_radius: Radius to consider for resource influence
        """
        # Build coefficients for the objective function (maximize overall resource utilization)
        c = []
        for m_type in self.mineral_types:
            for i in range(n_factions):
                # Get resource value at this faction's position
                x, y = faction_positions[i]
                field = self.pde_system.fields[m_type]
                local_value = float(
                    np.mean(
                        field[
                            max(0, x - influence_radius) : min(
                                self.width, x + influence_radius + 1
                            ),
                            max(0, y - influence_radius) : min(
                                self.height, y + influence_radius + 1
                            ),
                        ]
                    )
                )
                # Higher value means we want to prioritize this allocation
                c.append(-local_value)  # Negative because linprog minimizes

        # Variables are x_{i,j} = amount of mineral type i allocated to faction j
        # Total number of variables = n_minerals * n_factions
        n_vars = len(self.mineral_types) * n_factions

        # Constraint: sum of allocations for each mineral type across all factions <= total amount
        a_ub = []
        b_ub = []
        for m_idx, m_type in enumerate(self.mineral_types):
            # Build constraint row for this mineral type
            row = [0] * n_vars
            for f in range(n_factions):
                row[m_idx * n_factions + f] = 1.0
            a_ub.append(row)
            # Total available of this mineral type
            field = self.pde_system.fields[m_type]
            b_ub.append(float(np.sum(field)))

        # Additional optional constraints could be added here

        # Solve the linear program
        bounds = [(0, None) for _ in range(n_vars)]  # All variables non-negative
        result = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="simplex")

        # Results could be used to update the game state or inform other decisions
        # For now we're just computing the optimal allocation and not using it yet
        if result.success:
            return {
                m_type: {f: result.x[m_idx * n_factions + f] for f in range(n_factions)}
                for m_idx, m_type in enumerate(self.mineral_types)
            }
        return None
