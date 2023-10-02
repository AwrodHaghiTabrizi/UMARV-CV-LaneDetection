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

    model_ids = []
    models_dir = os.path.join(repo_dir, "models")
    if os.path.exists(models_dir):
        for model_dir_name in os.listdir(models_dir):
            if model_dir_name == "model_template":
                continue
            elif model_dir_name.startswith("model_"):
                model_id = model_dir_name.replace("model_", "")
                model_ids.append(model_id)

    performance_data = {}
    for model_id in model_ids:
        model_performance_dir = f"{repo_dir}/models/model_{model_id}/content/performance.json"
        if os.path.exists(model_performance_dir):
            with open(model_performance_dir, 'r') as performance_file:
                performance = json.load(performance_file)
                performance_data[model_id] = performance
        else:
            performance_data[model_id] = None

    benchmark_rankings = {}
    for model_id, performance in performance_data.items():
        if performance is not None:
            for benchmark, metrics in performance.items():
                if benchmark not in benchmark_rankings:
                    benchmark_rankings[benchmark] = []
                benchmark_rankings[benchmark].append((model_id, metrics[critiqueing_metric]))

    for benchmark, rankings in benchmark_rankings.items():
        benchmark_rankings[benchmark] = sorted(rankings, key=lambda x: x[1], reverse=True)

    print("\nModels without performance metrics:")
    no_models_without_metrics = True
    for model_id, performance in performance_data.items():
        if performance is None:
            no_models_without_metrics = False
            print(   f"{model_id} has no performance metrics.")
    if no_models_without_metrics:
        print("   None")

    print("\nModels missing metrics for specific benchmarks:")
    no_models_missing_metrics = True
    for model_id, performance in performance_data.items():
        if performance is not None:
            missing_benchmarks = [benchmark for benchmark in benchmark_rankings.keys() if benchmark not in performance]
            if missing_benchmarks:
                no_models_missing_metrics = False
                print(f"   {model_id} missing metrics for: {', '.join(missing_benchmarks)}")
    if no_models_missing_metrics:
        print("   None")

    for benchmark, rankings in benchmark_rankings.items():
        print(f"\nBest performing models for {benchmark}:")
        for rank, (model_id, mean_pixel_accuracy) in enumerate(rankings, start=1):
            print(f"   {rank}. {model_id} <> Mean Pixel Accuracy: {mean_pixel_accuracy*100:.2f} %")

if __name__ == "__main__":
    main()