#!/usr/bin/env python3
"""
Unit tests for the SymbioteEvolutionGenerator class.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock modules before importing entities
sys.modules["perlin_noise"] = MagicMock()
sys.modules["perlin_noise"].PerlinNoise = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.measure"] = MagicMock()

# Mock src modules
sys.modules["src.utils.logging_setup"] = MagicMock()
sys.modules["src.utils.logging_setup"].log_performance_start = MagicMock()
sys.modules["src.utils.logging_setup"].log_performance_end = MagicMock()
sys.modules["src.utils.logging_setup"].log_exception = MagicMock()
sys.modules["src.utils.logging_setup"].LogContext = MagicMock()

# Import the classes to test
from generators.symbiote_evolution_generator import SymbioteEvolutionGenerator
from entities.base_generator import BaseGenerator


class TestSymbioteEvolutionGenerator(unittest.TestCase):
    """Test cases for the SymbioteEvolutionGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for PerlinNoise
        self.perlin_mock = MagicMock()
        self.perlin_mock.return_value = 0.5

        # Patch the PerlinNoise class
        self.perlin_patcher = patch(
            "perlin_noise.PerlinNoise", return_value=self.perlin_mock
        )
        self.perlin_patcher.start()

        # Create a generator for testing
        self.generator = SymbioteEvolutionGenerator(
            entity_id="symb-123",
            seed=42,
            width=50,
            height=60,
            color=(50, 200, 150),
            position=(5, 10),
            initial_population=100,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        self.perlin_patcher.stop()

    def test_initialization(self):
        """Test that generator initializes with correct values."""
        # Test with explicit values
        self.assertEqual(self.generator.entity_id, "symb-123")
        self.assertEqual(self.generator.entity_type, "symbiote")
        self.assertEqual(self.generator.seed, 42)
        self.assertEqual(self.generator.width, 50)
        self.assertEqual(self.generator.height, 60)
        self.assertEqual(self.generator.color, (50, 200, 150))
        self.assertEqual(self.generator.position, (5, 10))
        self.assertEqual(self.generator.initial_population, 100)

        # Test inheritance
        self.assertIsInstance(self.generator, BaseGenerator)

    def test_default_initialization(self):
        """Test initialization with default values."""
        generator = SymbioteEvolutionGenerator()

        # Test default values
        self.assertEqual(generator.entity_type, "symbiote")
        self.assertEqual(generator.width, 100)
        self.assertEqual(generator.height, 100)
        self.assertEqual(generator.color, (50, 200, 150))
        self.assertIsNone(generator.position)
        self.assertEqual(generator.initial_population, 50)

        # Seed should be set to a random value
        self.assertIsNotNone(generator.seed)

    def test_generate_initial_population(self):
        """Test the generate_initial_population method."""
        # Generate initial population
        population = self.generator.generate_initial_population()

        # Verify the population size
        self.assertEqual(len(population), 100)

        # Test with different parameters
        generator = SymbioteEvolutionGenerator(initial_population=200)
        population = generator.generate_initial_population()
        self.assertEqual(len(population), 200)

    def test_generate_genome(self):
        """Test the generate_genome method."""
        # Generate a genome
        genome = self.generator.generate_genome()

        # Verify the genome is a dictionary
        self.assertIsInstance(genome, dict)

        # Verify the genome has the expected keys
        expected_keys = [
            "metabolism",
            "reproduction",
            "adaptation",
            "resistance",
            "symbiosis",
        ]
        for key in expected_keys:
            self.assertIn(key, genome)
            self.assertIsInstance(genome[key], (int, float))
            self.assertGreaterEqual(genome[key], 0)
            self.assertLessEqual(genome[key], 1)

        # Test with trait bias
        genome = self.generator.generate_genome(trait_bias={"metabolism": 0.8})
        self.assertGreaterEqual(genome["metabolism"], 0.5)  # Should be biased higher

    def test_evolve_population(self):
        """Test the evolve_population method."""
        # Create a simple population
        population = [self.generator.generate_genome() for _ in range(10)]

        # Evolve the population
        evolved_population = self.generator.evolve_population(
            population=population, generations=5, selection_pressure=0.5
        )

        # Verify the evolved population size
        self.assertEqual(len(evolved_population), 10)

        # Test with different parameters
        evolved_population = self.generator.evolve_population(
            population=population,
            generations=10,
            selection_pressure=0.7,
            mutation_rate=0.2,
        )
        self.assertEqual(len(evolved_population), 10)

    def test_select_parents(self):
        """Test the select_parents method."""
        # Create a simple population
        population = [self.generator.generate_genome() for _ in range(10)]

        # Select parents
        parents = self.generator._select_parents(
            population=population, selection_pressure=0.5
        )

        # Verify the parents list
        self.assertEqual(len(parents), 5)  # 50% of population

        # Test with different selection pressure
        parents = self.generator._select_parents(
            population=population, selection_pressure=0.3
        )
        self.assertEqual(len(parents), 3)  # 30% of population

    def test_crossover(self):
        """Test the crossover method."""
        # Create two parent genomes
        parent1 = {
            "metabolism": 0.8,
            "reproduction": 0.6,
            "adaptation": 0.4,
            "resistance": 0.5,
            "symbiosis": 0.7,
        }
        parent2 = {
            "metabolism": 0.2,
            "reproduction": 0.3,
            "adaptation": 0.9,
            "resistance": 0.6,
            "symbiosis": 0.4,
        }

        # Perform crossover
        child = self.generator._crossover(parent1, parent2)

        # Verify the child genome
        self.assertIsInstance(child, dict)
        for key in parent1.keys():
            self.assertIn(key, child)
            # Child traits should be between parent traits or equal to one of them
            self.assertTrue(
                min(parent1[key], parent2[key])
                <= child[key]
                <= max(parent1[key], parent2[key])
                or child[key] == parent1[key]
                or child[key] == parent2[key]
            )

    def test_mutate(self):
        """Test the mutate method."""
        # Create a genome
        genome = {
            "metabolism": 0.5,
            "reproduction": 0.5,
            "adaptation": 0.5,
            "resistance": 0.5,
            "symbiosis": 0.5,
        }

        # Create a copy for comparison
        original = genome.copy()

        # Mutate the genome
        mutated = self.generator._mutate(
            genome, mutation_rate=1.0
        )  # 100% mutation rate

        # Verify the mutated genome
        self.assertIsInstance(mutated, dict)
        for key in original.keys():
            self.assertIn(key, mutated)
            # With 100% mutation rate, at least some values should change

        # Test with 0% mutation rate (should not change)
        mutated = self.generator._mutate(genome, mutation_rate=0.0)
        for key in original.keys():
            self.assertEqual(mutated[key], original[key])

    def test_fitness(self):
        """Test the fitness method."""
        # Create a genome
        genome = {
            "metabolism": 0.8,
            "reproduction": 0.6,
            "adaptation": 0.4,
            "resistance": 0.5,
            "symbiosis": 0.7,
        }

        # Calculate fitness
        fitness = self.generator._fitness(genome)

        # Verify the fitness is a number
        self.assertIsInstance(fitness, (int, float))

        # Test with different environment factors
        fitness = self.generator._fitness(
            genome=genome,
            environment_factors={
                "resource_scarcity": 0.8,
                "predation_pressure": 0.6,
                "environmental_stability": 0.4,
            },
        )
        self.assertIsInstance(fitness, (int, float))

    def test_generate_symbiote_species(self):
        """Test the generate_symbiote_species method."""
        # Generate a symbiote species
        species = self.generator.generate_symbiote_species(
            evolution_steps=5, selection_pressure=0.5
        )

        # Verify the species
        self.assertIsInstance(species, dict)
        self.assertIn("population", species)
        self.assertIn("dominant_traits", species)
        self.assertIn("generation", species)

        # Verify the population
        self.assertEqual(len(species["population"]), 100)  # Initial population size

        # Test with different parameters
        species = self.generator.generate_symbiote_species(
            evolution_steps=10,
            selection_pressure=0.7,
            mutation_rate=0.2,
            environment_factors={
                "resource_scarcity": 0.8,
                "predation_pressure": 0.6,
                "environmental_stability": 0.4,
            },
        )
        self.assertIsInstance(species, dict)
        self.assertEqual(len(species["population"]), 100)
        self.assertEqual(species["generation"], 10)

    def test_to_dict(self):
        """Test the to_dict method."""
        # Convert generator to dictionary
        generator_dict = self.generator.to_dict()

        # Verify the dictionary
        self.assertIsInstance(generator_dict, dict)
        self.assertEqual(generator_dict["entity_id"], "symb-123")
        self.assertEqual(generator_dict["entity_type"], "symbiote")
        self.assertEqual(generator_dict["seed"], 42)
        self.assertEqual(generator_dict["width"], 50)
        self.assertEqual(generator_dict["height"], 60)
        self.assertEqual(generator_dict["color"], (50, 200, 150))
        self.assertEqual(generator_dict["position"], (5, 10))
        self.assertEqual(generator_dict["initial_population"], 100)

    def test_from_dict(self):
        """Test the from_dict method."""
        # Create a dictionary
        generator_dict = {
            "entity_id": "symb-456",
            "entity_type": "symbiote",
            "seed": 123,
            "width": 80,
            "height": 90,
            "color": (60, 180, 120),
            "position": (15, 25),
            "initial_population": 150,
        }

        # Create generator from dictionary
        generator = SymbioteEvolutionGenerator.from_dict(generator_dict)

        # Verify the generator
        self.assertEqual(generator.entity_id, "symb-456")
        self.assertEqual(generator.entity_type, "symbiote")
        self.assertEqual(generator.seed, 123)
        self.assertEqual(generator.width, 80)
        self.assertEqual(generator.height, 90)
        self.assertEqual(generator.color, (60, 180, 120))
        self.assertEqual(generator.position, (15, 25))
        self.assertEqual(generator.initial_population, 150)


if __name__ == "__main__":
    unittest.main()
