#!/usr/bin/env python3
"""
Main entry point for the comprehensive GPU benchmarking script.
This file provides command-line interface to run the benchmark functions.
"""

import os
import sys
import argparse
import logging
from benchmark_comprehensive_gpu import (
    run_all_benchmarks,
    benchmark_cellular_automaton,
    benchmark_noise_generation,
    benchmark_clustering,
    benchmark_value_generation,
    benchmark_memory_transfer,
    visualize_benchmark_results
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive GPU Benchmarking for Space Muck")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./benchmark_results",
        help="Directory to save benchmark results and visualizations"
    )
    parser.add_argument(
        "--small-grid-sizes", 
        type=int, 
        nargs="+", 
        default=[32, 64, 128, 256, 512, 1024],
        help="Grid sizes for 2D operations (cellular automaton, noise)"
    )
    parser.add_argument(
        "--large-data-sizes", 
        type=int, 
        nargs="+", 
        default=[1000, 10000, 100000, 1000000],
        help="Data sizes for 1D operations (clustering, memory transfer)"
    )
    parser.add_argument(
        "--repetitions", 
        type=int, 
        default=3,
        help="Number of repetitions for each benchmark"
    )
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["all", "ca", "noise", "clustering", "value", "memory"],
        default="all",
        help="Specific benchmark to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Run specific benchmark or all benchmarks
    if args.benchmark == "all":
        run_all_benchmarks(
            output_dir=args.output_dir,
            small_grid_sizes=args.small_grid_sizes,
            large_data_sizes=args.large_data_sizes,
            repetitions=args.repetitions
        )
    elif args.benchmark == "ca":
        # Run cellular automaton benchmark
        logging.info("\n===== CELLULAR AUTOMATON BENCHMARKS =====")
        ca_results = benchmark_cellular_automaton(
            grid_sizes=args.small_grid_sizes,
            iterations=5,
            repetitions=args.repetitions
        )
        visualize_benchmark_results(ca_results, "Cellular Automaton", args.output_dir)
    elif args.benchmark == "noise":
        # Run noise generation benchmark
        logging.info("\n===== NOISE GENERATION BENCHMARKS =====")
        noise_results = benchmark_noise_generation(
            grid_sizes=args.small_grid_sizes,
            octaves=4,
            repetitions=args.repetitions
        )
        visualize_benchmark_results(noise_results, "Noise Generation", args.output_dir)
    elif args.benchmark == "clustering":
        # Run clustering benchmark
        logging.info("\n===== CLUSTERING BENCHMARKS =====")
        clustering_results = benchmark_clustering(
            data_sizes=args.large_data_sizes[:3],  # Use smaller subset for clustering
            n_clusters=5,
            repetitions=args.repetitions
        )
        visualize_benchmark_results(clustering_results, "Clustering", args.output_dir)
    elif args.benchmark == "value":
        # Run value generation benchmark
        logging.info("\n===== VALUE GENERATION BENCHMARKS =====")
        value_results = benchmark_value_generation(
            grid_sizes=args.small_grid_sizes,
            repetitions=args.repetitions
        )
        visualize_benchmark_results(value_results, "Value Generation", args.output_dir)
    elif args.benchmark == "memory":
        # Run memory transfer benchmark
        logging.info("\n===== MEMORY TRANSFER BENCHMARKS =====")
        memory_results = benchmark_memory_transfer(
            data_sizes=args.large_data_sizes,
            repetitions=args.repetitions
        )
        visualize_benchmark_results(memory_results, "Memory Transfer", args.output_dir)
