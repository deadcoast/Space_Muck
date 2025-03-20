"""
pareto_optimizer.py

Implements advanced multi-objective optimization solvers that generate Pareto-optimal
solutions for complex tradeoffs in the Space Muck economy. This allows for finding
solutions that balance competing objectives like minimizing aggression and maximizing profit
without having to predefine weights.
"""

import importlib.util
import numpy as np
import scipy.optimize
from typing import Dict, List, Optional, Callable
import matplotlib.pyplot as plt
from src.algorithms.economy_decision import MineralResourceOptimizer

# Check if pygame is available for visualization using importlib.util (per linting recommendation)
PYGAME_AVAILABLE = importlib.util.find_spec("pygame") is not None


class ParetoOptimalSolution:
    """
    Data class for storing Pareto-optimal solutions with their objective values.
    """
    
    def __init__(self, solution: Dict[str, float], objective_values: Dict[str, float]):
        """
        Initialize a Pareto-optimal solution.
        
        Args:
            solution: Dictionary containing decision variables (e.g., feeding amounts)
            objective_values: Dictionary mapping objective names to their values
        """
        self.solution = solution
        self.objective_values = objective_values
    
    def __repr__(self) -> str:
        """String representation of the solution."""
        return f"ParetoOptimalSolution(objectives={self.objective_values})"


class MultiObjectiveOptimizer:
    """
    Advanced optimizer that handles multiple competing objectives, such as
    profit maximization and aggression minimization, to generate a Pareto front
    of optimal solutions.
    
    This allows decision-makers to select from a range of optimal tradeoffs
    rather than predefining weights for objectives.
    """
    
    def __init__(self, mineral_types: List[str]):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            mineral_types: List of mineral type identifiers
        """
        self.mineral_types = mineral_types
        self.base_optimizer = MineralResourceOptimizer(mineral_types)
    
    def _calculate_appeasement_bounds(
        self, 
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None
    ) -> tuple[float, float]:
        """
        Calculate the minimum and maximum possible appeasement values.
        
        Args:
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            min_feeding_requirements: Minimum feeding requirements by mineral type
            
        Returns:
            Tuple of (min_appeasement, max_appeasement)
        """
        # Maximize appeasement (feed everything)
        max_appease_solution = {m: total_minerals.get(m, 0.0) for m in self.mineral_types}
        max_appeasement = sum(max_appease_solution[m] * feed_benefits.get(m, 0.0) for m in self.mineral_types)
        
        # Minimize appeasement (feed only required minerals)
        min_appease_solution = {
            m: min_feeding_requirements.get(m, 0.0) if min_feeding_requirements else 0.0 
            for m in self.mineral_types
        }
        min_appeasement = sum(min_appease_solution[m] * feed_benefits.get(m, 0.0) for m in self.mineral_types)
        
        return min_appeasement, max_appeasement
    
    def _setup_optimization_problem(
        self,
        appeasement_level: float,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None
    ) -> tuple:
        """
        Set up the optimization problem for a specific appeasement level.
        
        Args:
            appeasement_level: Target appeasement level for this optimization
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            sell_values: Dictionary of {mineral_type: profit_per_unit_sold}
            min_feeding_requirements: Minimum feeding requirements by mineral type
            
        Returns:
            Tuple of (mineral_order, bounds, constraints, x0, objective function)
        """
        n = len(self.mineral_types)
        mineral_order = list(self.mineral_types)
        
        # Define objective with properly captured variables to prevent closure issues
        def objective(x, mineral_types=mineral_order, num_minerals=n):
            # x is array of feeding amounts, we want to maximize profit (minimize negative profit)
            feeding = {mineral_types[i]: x[i] for i in range(num_minerals)}
            # Calculate selling amounts and profit
            selling = {m: total_minerals.get(m, 0.0) - feeding[m] for m in mineral_types}
            profit = sum(selling[m] * sell_values.get(m, 0.0) for m in mineral_types)
            return -profit  # Negate for minimization
        
        # Define constraint with properly captured variables to prevent closure issues
        def appeasement_constraint(x, mineral_types=mineral_order, num_minerals=n, target_level=appeasement_level):
            # Constraint: appeasement >= appeasement_level
            feeding = {mineral_types[i]: x[i] for i in range(num_minerals)}
            actual_appeasement = sum(feeding[m] * feed_benefits.get(m, 0.0) for m in mineral_types)
            return actual_appeasement - target_level
        
        # Bounds for each mineral: [0, total_amount]
        bounds = [(0, total_minerals.get(m, 0.0)) for m in mineral_order]
        
        # Set up feeding requirement constraints
        constraints = [{'type': 'ineq', 'fun': appeasement_constraint}]
        
        # Add minimum feeding constraints if needed
        if min_feeding_requirements:
            for i, m in enumerate(mineral_order):
                if m in min_feeding_requirements:
                    # Create a function with parameters to avoid closure issues
                    def create_min_feeding_constraint(mineral_idx, requirement):
                        def constraint_function(x):
                            return x[mineral_idx] - requirement
                        return constraint_function
                    
                    # Create a constraint function with properly captured variables
                    min_feeding_constraint = create_min_feeding_constraint(i, min_feeding_requirements[m])
                    constraints.append({'type': 'ineq', 'fun': min_feeding_constraint})
        
        # Initial guess: proportional feeding based on benefits
        total_benefit = sum(feed_benefits.get(m, 0.0) for m in mineral_order)
        if total_benefit > 0:
            x0 = [
                appeasement_level * feed_benefits.get(m, 0.0) / total_benefit 
                if feed_benefits.get(m, 0.0) > 0 else 0.0 
                for m in mineral_order
            ]
        else:
            x0 = [0.0] * n
        
        return mineral_order, bounds, constraints, x0, objective
    
    def _extract_solution(
        self,
        result,
        mineral_order: List[str],
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float]
    ) -> Optional[ParetoOptimalSolution]:
        """
        Extract a solution from an optimization result.
        
        Args:
            result: Optimization result from scipy.optimize.minimize
            mineral_order: List of mineral types in the order used for optimization
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            sell_values: Dictionary of {mineral_type: profit_per_unit_sold}
            
        Returns:
            ParetoOptimalSolution object or None if optimization failed
        """
        if not result.success:
            return None
            
        n = len(mineral_order)
        # Convert back to our dictionary format
        solution = {mineral_order[i]: result.x[i] for i in range(n)}
        
        # Calculate objective values
        selling = {m: total_minerals.get(m, 0.0) - solution[m] for m in mineral_order}
        profit = sum(selling[m] * sell_values.get(m, 0.0) for m in mineral_order)
        appeasement = sum(solution[m] * feed_benefits.get(m, 0.0) for m in mineral_order)
        
        return ParetoOptimalSolution(
            solution=solution,
            objective_values={
                'profit': profit,
                'appeasement': appeasement
            }
        )
    
    def epsilon_constraint_method(
        self,
        total_minerals: Dict[str, float],
        feed_benefits: Dict[str, float],
        sell_values: Dict[str, float],
        min_feeding_requirements: Dict[str, float] = None,
        num_points: int = 10
    ) -> List[ParetoOptimalSolution]:
        """
        Generate Pareto-optimal solutions using the epsilon-constraint method.
        
        This method optimizes one objective (profit) while constraining the other
        (appeasement/aggression) to different levels, generating a set of non-dominated
        solutions that represent the best possible tradeoffs.
        
        Args:
            total_minerals: Dictionary of {mineral_type: amount_available}
            feed_benefits: Dictionary of {mineral_type: appeasement_gained_per_unit}
            sell_values: Dictionary of {mineral_type: profit_per_unit_sold}
            min_feeding_requirements: Minimum feeding requirements by mineral type
            num_points: Number of points to generate on the Pareto front
            
        Returns:
            List of ParetoOptimalSolution objects representing the Pareto front
        """
        pareto_front = []
        
        # Calculate the min and max appeasement bounds
        min_appeasement, max_appeasement = self._calculate_appeasement_bounds(
            total_minerals, feed_benefits, min_feeding_requirements
        )
        
        # Generate points along the Pareto front by varying the appeasement constraint
        appeasement_levels = np.linspace(min_appeasement, max_appeasement, num_points)
        
        for appeasement_level in appeasement_levels:
            # Set up optimization problem for this appeasement level
            mineral_order, bounds, constraints, x0, objective = self._setup_optimization_problem(
                appeasement_level, total_minerals, feed_benefits, sell_values, min_feeding_requirements
            )
            
            # Apply scipy's optimizer
            result = scipy.optimize.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
            
            # Extract solution if optimization was successful
            solution = self._extract_solution(
                result, mineral_order, total_minerals, feed_benefits, sell_values
            )
            
            if solution is not None:
                pareto_front.append(solution)
        
        return pareto_front
    
    def visualize_pareto_front(
        self, 
        pareto_solutions: List[ParetoOptimalSolution],
        title: str = "Pareto Front: Profit vs. Appeasement"
    ) -> None:
        """
        Visualize the Pareto front for the profit vs. appeasement tradeoff.
        
        Args:
            pareto_solutions: List of ParetoOptimalSolution objects
            title: Chart title
        """
        if not PYGAME_AVAILABLE:
            print("Pygame is not available for visualization.")
            return
            
        profits = [sol.objective_values['profit'] for sol in pareto_solutions]
        appeasements = [sol.objective_values['appeasement'] for sol in pareto_solutions]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(profits, appeasements, c='blue', s=50)
        plt.plot(profits, appeasements, 'b--')
        
        plt.title(title)
        plt.xlabel("Profit")
        plt.ylabel("Appeasement")
        plt.grid(True)
        
        # Add annotation for a few selected points
        for i in range(0, len(pareto_solutions), max(1, len(pareto_solutions) // 5)):
            sol = pareto_solutions[i]
            plt.annotate(
                f"({sol.objective_values['profit']:.1f}, {sol.objective_values['appeasement']:.1f})",
                (sol.objective_values['profit'], sol.objective_values['appeasement']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
        
        plt.tight_layout()
        plt.show()
    
    def find_solution_with_desired_tradeoff(
        self,
        pareto_solutions: List[ParetoOptimalSolution],
        preference_function: Callable[[Dict[str, float]], float]
    ) -> Optional[ParetoOptimalSolution]:
        """
        Find the Pareto-optimal solution that best matches the decision maker's preferences.
        
        Args:
            pareto_solutions: List of ParetoOptimalSolution objects
            preference_function: Function that takes objective values and returns a
                                preference score (higher is better)
                                
        Returns:
            The ParetoOptimalSolution with the highest preference score
        """
        if not pareto_solutions:
            return None
            
        # Calculate preference scores for each solution
        scores = [preference_function(sol.objective_values) for sol in pareto_solutions]
        
        # Find the solution with the highest score
        best_index = np.argmax(scores)
        return pareto_solutions[best_index]
    
    def weighted_sum_preference(
        self,
        profit_weight: float,
        appeasement_weight: float
    ) -> Callable[[Dict[str, float]], float]:
        """
        Create a weighted sum preference function for finding solutions.
        
        Args:
            profit_weight: Weight for profit objective
            appeasement_weight: Weight for appeasement objective
            
        Returns:
            Preference function that can be used with find_solution_with_desired_tradeoff
        """
        def preference(objective_values: Dict[str, float]) -> float:
            return (profit_weight * objective_values.get('profit', 0.0) +
                   appeasement_weight * objective_values.get('appeasement', 0.0))
        
        return preference
