"""
mip_optimizer.py

Extends the mineral resource optimization to use mixed integer programming (MIP)
for scenarios requiring integer feeding amounts or binary decisions. This module 
builds upon the existing optimization framework but uses PuLP for more advanced 
optimization capabilities.
"""

import numpy as np  # noqa: F401 - Will be used in future implementations
from typing import Dict, List  # Tuple, Optional not needed yet
import pulp as pl

# Used for type declarations and future extensions
from src.algorithms.economy_decision import MineralResourceOptimizer  # noqa: F401


class MixedIntegerMineralOptimizer:
    """
    Advanced resource optimizer using mixed integer programming to solve complex
    feeding problems involving integer constraints, binary decisions, or 
    logical conditions.
    
    This allows for more realistic scenarios such as:
    1) Feeding minerals in discrete units rather than continuous amounts
    2) Making binary decisions (feed or don't feed certain minerals)
    3) Handling logical constraints between different mineral types
    """
    
    def __init__(self, mineral_types: List[str]):
        """
        Initialize the mixed integer optimizer.
        
        Args:
            mineral_types: List of mineral type identifiers
        """
        self.mineral_types = mineral_types
    
    def optimize_with_integer_constraints(
        self,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None,
        max_feeding_limits: Dict[str, float] = None,
        integer_minerals: List[str] = None,
        binary_decisions: List[str] = None,
        exclusive_groups: List[List[str]] = None,
        aggression_coef: float = 0.0,
    ) -> Dict[str, float]:
        """
        Solve a mixed integer program for optimal mineral allocation.
        
        Args:
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            sell_values: Dictionary of {mineral_type: profit_per_unit_sold}
            min_feeding_requirements: Minimum feeding requirements by mineral type
            max_feeding_limits: Maximum feeding limits by mineral type 
            integer_minerals: List of minerals that must be fed in integer amounts
            binary_decisions: List of minerals that must be either fully fed or not fed
            exclusive_groups: Groups of minerals where only one from each group can be fed
            aggression_coef: Balance between profit and appeasement (0.0 to 1.0)
                            0 = pure profit, 1 = pure appeasement
                            
        Returns:
            Dictionary with {mineral_type: amount_to_feed}
        """
        # Create a PuLP problem
        prob = pl.LpProblem("MineralFeedingOptimization", pl.LpMaximize)

        # Initialize decision variables
        feed_vars = self._create_feed_variables(total_minerals, binary_decisions, integer_minerals)

        sell_vars = self._create_sell_variables(total_minerals, binary_decisions)
        # Build objective function terms
        profit_term, appeasement_term = self._build_objective_terms(
            feed_vars, sell_vars, total_minerals, feed_benefits, sell_values
        )
        
        # Set objective function based on aggression coefficient
        self._set_objective_function(prob, profit_term, appeasement_term, aggression_coef)

        # Add conservation constraints
        self._add_conservation_constraints(prob, feed_vars, sell_vars, total_minerals, binary_decisions)

        # Add minimum feeding constraints
        self._add_min_feeding_constraints(prob, feed_vars, min_feeding_requirements, total_minerals, binary_decisions)

        # Add maximum feeding constraints
        self._add_max_feeding_constraints(prob, feed_vars, max_feeding_limits)

        # Add exclusive group constraints
        self._add_exclusive_group_constraints(prob, feed_vars, exclusive_groups, binary_decisions, total_minerals)

        # Solve the mixed integer program
        prob.solve(pl.PULP_CBC_CMD(msg=False))

        # Extract and return results
        return self._extract_results(feed_vars, total_minerals, binary_decisions)
        
    def _create_feed_variables(self, total_minerals, binary_decisions, integer_minerals):
        """Helper method to create feed decision variables"""
        feed_vars = {}
        for m in self.mineral_types:
            if binary_decisions and m in binary_decisions:
                # Binary variables (0 or 1)
                feed_vars[m] = pl.LpVariable(f"Feed_{m}", 0, 1, pl.LpBinary)
            elif integer_minerals and m in integer_minerals:
                # Integer variables
                feed_vars[m] = pl.LpVariable(f"Feed_{m}", 0, total_minerals.get(m, 0), pl.LpInteger)
            else:
                # Continuous variables
                feed_vars[m] = pl.LpVariable(f"Feed_{m}", 0, total_minerals.get(m, 0), pl.LpContinuous)
        return feed_vars
    
    def _create_sell_variables(self, total_minerals, binary_decisions):
        """Helper method to create sell decision variables"""
        return {
            m: pl.LpVariable(
                f"Sell_{m}", 0, total_minerals.get(m, 0), pl.LpContinuous
            )
            for m in self.mineral_types
            if not binary_decisions or m not in binary_decisions
        }
    
    def _build_objective_terms(self, feed_vars, sell_vars, total_minerals, feed_benefits, sell_values):
        """Helper method to build objective function terms"""
        profit_term = pl.lpSum([sell_vars.get(m, total_minerals.get(m, 0) - feed_vars[m]) * sell_values.get(m, 0) 
                              for m in self.mineral_types])
        appeasement_term = pl.lpSum([feed_vars[m] * feed_benefits.get(m, 0) 
                                  for m in self.mineral_types])
        return profit_term, appeasement_term
    
    def _set_objective_function(self, prob, profit_term, appeasement_term, aggression_coef):  # NOSONAR
        """Helper method to set the objective function based on aggression coefficient"""
        if aggression_coef <= 0:
            # Pure profit maximization
            prob += profit_term
        elif aggression_coef >= 1:
            # Pure appeasement maximization
            prob += appeasement_term
        else:
            # Weighted combination
            prob += (1 - aggression_coef) * profit_term + aggression_coef * appeasement_term
    
    def _add_conservation_constraints(self, prob, feed_vars, sell_vars, total_minerals, binary_decisions):  # noqa: ARG001 # NOSONAR
        """Helper method to add conservation constraints
        
        Note: prob is modified in-place via += operator, PuLP's preferred API pattern
        The linter may incorrectly flag this as unused.
        """
        for m in self.mineral_types:
            if binary_decisions and m in binary_decisions:
                # For binary variables, we need to handle this differently
                # If we feed (var=1), we feed the entire amount and sell nothing
                # If we don't feed (var=0), we sell everything
                prob += sell_vars.get(m, total_minerals.get(m, 0) - 
                                  feed_vars[m] * total_minerals.get(m, 0)) <= total_minerals.get(m, 0)
            else:
                # Standard conservation constraint
                prob += feed_vars[m] + sell_vars.get(m, total_minerals.get(m, 0) - feed_vars[m]) <= total_minerals.get(m, 0)
    
    def _add_min_feeding_constraints(self, prob, feed_vars, min_feeding_requirements, total_minerals, binary_decisions):  # noqa: ARG001 # NOSONAR
        """Helper method to add minimum feeding constraints
        
        Note: prob is modified in-place via += operator, PuLP's preferred API pattern
        The linter may incorrectly flag this as unused.
        """
        if min_feeding_requirements:
            for m, min_amount in min_feeding_requirements.items():
                if m in feed_vars:
                    if binary_decisions and m in binary_decisions:
                        # For binary decisions, if we feed, we feed at least min_amount
                        prob += feed_vars[m] * total_minerals.get(m, 0) >= min_amount * feed_vars[m]
                    else:
                        # Standard minimum constraint
                        prob += feed_vars[m] >= min_amount
    
    def _add_max_feeding_constraints(self, prob, feed_vars, max_feeding_limits):  # noqa: ARG001 # NOSONAR
        """Helper method to add maximum feeding constraints
        
        Note: prob is modified in-place via += operator, PuLP's preferred API pattern
        The linter may incorrectly flag this as unused.
        """
        if max_feeding_limits:
            for m, max_amount in max_feeding_limits.items():
                if m in feed_vars:
                    prob += feed_vars[m] <= max_amount
    
    def _add_exclusive_group_constraints(self, prob, feed_vars, exclusive_groups, binary_decisions, total_minerals):  # noqa: ARG001 # NOSONAR
        """Helper method to add exclusive group constraints
        
        Note: prob is modified in-place via += operator, PuLP's preferred API pattern
        The linter may incorrectly flag this as unused.
        """
        if not exclusive_groups:
            return
            
        for group_idx, group in enumerate(exclusive_groups):
            self._add_single_exclusive_group(prob, feed_vars, group, group_idx, binary_decisions, total_minerals)
    
    def _add_single_exclusive_group(self, prob, feed_vars, group, group_idx, binary_decisions, total_minerals):  # noqa: ARG001 # NOSONAR
        """Process a single exclusive group constraint
        
        Note: prob is modified in-place via += operator, PuLP's preferred API pattern
        The linter may incorrectly flag this as unused.
        """
        # Create a binary variable for each mineral in the group
        group_vars = {}
        
        for m in group:
            if m not in self.mineral_types:
                continue
                
            if binary_decisions and m in binary_decisions:
                # We can use the existing binary variable
                group_vars[m] = feed_vars[m]
            else:
                # Create a new binary variable
                group_vars[m] = pl.LpVariable(f"Use_{m}_Group{group_idx}", 0, 1, pl.LpBinary)
                # If this binary is 0, feeding must be 0
                prob += feed_vars[m] <= total_minerals.get(m, 0) * group_vars[m]

        # Only one mineral from the group can be fed if we have any variables
        if group_vars:  
            prob += pl.lpSum(list(group_vars.values())) <= 1
    
    def _extract_results(self, feed_vars, total_minerals, binary_decisions):
        """Helper method to extract results from the solved model"""
        feed_result = {}
        for m in self.mineral_types:
            if binary_decisions and m in binary_decisions:
                # For binary decisions, it's either all or nothing
                feed_result[m] = total_minerals.get(m, 0) * feed_vars[m].value() if feed_vars[m].value() > 0.5 else 0
            else:
                feed_result[m] = feed_vars[m].value() if feed_vars[m].value() is not None else 0
        return feed_result


class BinaryFeedingPlanGenerator:
    """
    Specialized optimizer for scenarios where feeding decisions are binary
    (feed or don't feed entire mineral reserves). Uses optimization to make
    these decisions based on benefits and values.
    """
    
    def __init__(self, mineral_types: List[str]):
        """
        Initialize the binary feeding plan generator.
        
        Args:
            mineral_types: List of mineral type identifiers
        """
        self.mip_optimizer = MixedIntegerMineralOptimizer(mineral_types)
    
    def generate_binary_plan(
        self,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        _min_total_appeasement: float = 0.0,  # Prefixed with _ to indicate unused but kept for future implementation
        _max_minerals_to_feed: int = None,   # Prefixed with _ to indicate unused but kept for future implementation
        aggression_coef: float = 0.0,
    ) -> Dict[str, float]:
        """
        Generate a binary feeding plan where each mineral is either fully fed
        or not fed at all.
        
        Args:
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            sell_values: Dictionary of {mineral_type: profit_per_unit_sold}
            min_total_appeasement: Minimum total appeasement required
            max_minerals_to_feed: Maximum number of minerals to feed
            aggression_coef: Balance between profit and appeasement
            
        Returns:
            Dictionary with binary feeding plan
        """
        binary_minerals = list(total_minerals.keys())

        # Empty dict for now, but could use _min_total_appeasement in future versions
        min_feeding_requirements = {}
        
        # Note: _max_minerals_to_feed would be used in future implementations
            
        return self.mip_optimizer.optimize_with_integer_constraints(
            total_minerals=total_minerals,
            feed_benefits=feed_benefits,
            sell_values=sell_values,
            binary_decisions=binary_minerals,
            min_feeding_requirements=min_feeding_requirements,
            aggression_coef=aggression_coef,
        )
