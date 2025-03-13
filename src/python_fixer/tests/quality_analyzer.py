#!/usr/bin/env python3

# -----------------------------
# QUALITY ANALYZER
# -----------------------------
#
# Parent: analysis.tests
# Dependencies: networkx, radon, typing, logging
#
# MAP: /project_root/analysis/tests
# EFFECT: Analyzes code quality and dependencies
# NAMING: Quality[Type]Analyzer

class QualityAnalyzer:
    """Analyzes code quality and dependencies.

    This system provides:
    1. Circular dependency detection
    2. Code complexity analysis
    3. Performance impact assessment
    4. Code quality metrics
    """

# Standard library imports
import logging

# Third-party library imports

# Local application imports
from radon.complexity import cc_visit
from radon.metrics import h_visit
from typing import Any, Dict, List
import networkx as nx

    def __init__(self):
        """Initialize the quality analyzer."""
        self.logger = logging.getLogger(__name__)
        self.dependency_graph = nx.DiGraph()
        self.complexity_scores = {}  # type: Dict[str, float]
        self.quality_metrics = {}  # type: Dict[str, Dict[str, float]]

    def analyze_dependencies(
        self, module_dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """Analyze module dependencies for circular references.

        Args:
            module_dependencies: Dictionary mapping modules to their dependencies

        Returns:
            List of circular dependency chains
        """
        # Build dependency graph
        self.dependency_graph.clear()
        for module, deps in module_dependencies.items():
            self.dependency_graph.add_node(module)
            for dep in deps:
                self.dependency_graph.add_edge(module, dep)

        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                for cycle in cycles:
                    self.logger.warning(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
            return cycles
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {str(e)}")
            return []

    def analyze_complexity(self, source_code: str, module_name: str) -> float:
        """Analyze code complexity using Radon.

        Args:
            source_code: Source code to analyze
            module_name: Name of the module being analyzed

        Returns:
            Cyclomatic complexity score
        """
        try:
            # Calculate cyclomatic complexity
            complexity_results = cc_visit(source_code)
            total_complexity = sum(result.complexity for result in complexity_results)

            # Store results
            self.complexity_scores[module_name] = total_complexity

            # Log warnings for high complexity
            if total_complexity > 50:
                self.logger.warning(
                    f"High complexity in {module_name}: {total_complexity}"
                )

            return total_complexity

        except Exception as e:
            self.logger.error(f"Error analyzing complexity for {module_name}: {str(e)}")
            return 0.0

    def analyze_quality_metrics(
        self, source_code: str, module_name: str
    ) -> Dict[str, float]:
        """Analyze code quality metrics using Radon.

        Args:
            source_code: Source code to analyze
            module_name: Name of the module being analyzed

        Returns:
            Dictionary of quality metrics
        """
        try:
            # Calculate Halstead metrics
            halstead_metrics = h_visit(source_code)

            metrics = {
                "volume": halstead_metrics.total.volume,
                "difficulty": halstead_metrics.total.difficulty,
                "effort": halstead_metrics.total.effort,
                "bugs": halstead_metrics.total.bugs,
                "time": halstead_metrics.total.time,
            }

            # Store results
            self.quality_metrics[module_name] = metrics

            # Log warnings for concerning metrics
            if metrics["bugs"] > 0.5:
                self.logger.warning(
                    f"High bug potential in {module_name}: {metrics['bugs']:.2f}"
                )

            if metrics["difficulty"] > 30:
                self.logger.warning(
                    f"High difficulty in {module_name}: {metrics['difficulty']:.2f}"
                )

            return metrics

        except Exception as e:
            self.logger.error(
                f"Error analyzing quality metrics for {module_name}: {str(e)}"
            )
            return {}

    def analyze_performance_impact(
        self, original_code: str, enhanced_code: str, module_name: str
    ) -> Dict[str, float]:
        """Analyze performance impact of enhancements.

        Args:
            original_code: Original source code
            enhanced_code: Enhanced source code
            module_name: Name of the module being analyzed

        Returns:
            Dictionary of impact metrics
        """
        try:
            # Calculate metrics for both versions
            original_metrics = self.analyze_quality_metrics(
                original_code, f"{module_name}_original"
            )
            enhanced_metrics = self.analyze_quality_metrics(
                enhanced_code, f"{module_name}_enhanced"
            )

            # Calculate impact percentages
            impact = {
                metric: (
                    (
                        (enhanced_metrics[metric] - original_metrics[metric])
                        / original_metrics[metric]
                        * 100
                    )
                    if original_metrics[metric] > 0
                    else 0.0
                )
                for metric in original_metrics
            }

            # Log significant impacts
            for metric, change in impact.items():
                if abs(change) > 20:
                    level = "warning" if change > 0 else "info"
                    getattr(self.logger, level)(
                        f"Significant {metric} impact in {module_name}: {change:+.1f}%"
                    )

            return impact

        except Exception as e:
            self.logger.error(
                f"Error analyzing performance impact for {module_name}: {str(e)}"
            )
            return {}

    def get_module_quality_report(self, module_name: str) -> Dict[str, Any]:
        """Get a comprehensive quality report for a module.

        Args:
            module_name: Name of the module to report on

        Returns:
            Dictionary containing all quality metrics
        """
        return {
            "complexity": self.complexity_scores.get(module_name, 0.0),
            "metrics": self.quality_metrics.get(module_name, {}),
            "dependencies": (
                list(self.dependency_graph.successors(module_name))
                if module_name in self.dependency_graph
                else []
            ),
        }

    def clear_analysis(self) -> None:
        """Clear all analysis results."""
        self.dependency_graph.clear()
        self.complexity_scores.clear()
        self.quality_metrics.clear()
