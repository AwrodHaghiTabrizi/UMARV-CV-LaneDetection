import os
import shutil
from datetime import datetime
import logging
import sys
import json
import nbformat
from operator import itemgetter
from collections import defaultdict

repo_dir = os.getcwd()

src_dir = os.path.join(repo_dir, "src")
sys.path.insert(0, src_dir)
from helpers import *

def main():

    critiqueing_metric = "Mean Pixel Accuracy"

    algorithm_ids = []
    algorithms_dir = f"{repo_dir}/algorithms"
    if os.path.exists(algorithms_dir):
        for algorithm_dir_name in os.listdir(algorithms_dir):
            if algorithm_dir_name == "algorithm_template":
                continue
            elif algorithm_dir_name.startswith("algorithm_"):
                algorithm_id = algorithm_dir_name.replace("algorithm_", "")
                algorithm_ids.append(algorithm_id)

    performance_data = {}
    for algorithm_id in algorithm_ids:
        algorithm_performance_dir = os.path.join(algorithms_dir, f"algorithm_{algorithm_id}/content/performance.json")
        if os.path.exists(algorithm_performance_dir):
            with open(algorithm_performance_dir, 'r') as performance_file:
                performance = json.load(performance_file)
                performance_data[f"algorithm_{algorithm_id}"] = performance
        else:
            performance_data[f"algorithm_{algorithm_id}"] = None

    benchmark_rankings = {}
    for algorithm_id, performance in performance_data.items():
        if performance is not None:
            for benchmark, metrics in performance.items():
                if benchmark not in benchmark_rankings:
                    benchmark_rankings[benchmark] = []
                benchmark_rankings[benchmark].append((algorithm_id, metrics[critiqueing_metric]))

    for benchmark, rankings in benchmark_rankings.items():
        benchmark_rankings[benchmark] = sorted(rankings, key=lambda x: x[1], reverse=True)

    print("\nAlgorithms without performance metrics:")
    no_algorithms_without_metrics = True
    for algorithm_id, performance in performance_data.items():
        if performance is None:
            no_algorithms_without_metrics = False
            print(f"   {algorithm_id} has no performance metrics.")
    if no_algorithms_without_metrics:
        print("   None")

    print("\nAlgorithms missing metrics for specific benchmarks:")
    no_algorithms_missing_metrics = True
    for algorithm_id, performance in performance_data.items():
        if performance is not None:
            missing_benchmarks = [benchmark for benchmark in benchmark_rankings.keys() if benchmark not in performance]
            if missing_benchmarks:
                no_algorithms_missing_metrics = False
                print(f"   {algorithm_id} missing metrics for: {', '.join(missing_benchmarks)}")
    if no_algorithms_missing_metrics:
        print("\tNone")

    for benchmark, rankings in benchmark_rankings.items():
        print(f"\nBest performing algorithms for {benchmark}:")
        for rank, (algorithm_id, mean_pixel_accuracy) in enumerate(rankings, start=1):
            print(f"   {rank}. {algorithm_id} <> Mean Pixel Accuracy: {mean_pixel_accuracy * 100:.2f} %")

if __name__ == "__main__":
    main()