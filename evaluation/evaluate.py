import os
import subprocess
import sys
import pandas as pd
from evaluation.evaluation_utils import load_ground_truth, calculate_mrr, calculate_hit_rate
from src.retrieval.search import PGSearcher


def run_ingestion(use_chunking: bool):
    """Executes ingestion script in a subprocess to refresh database state."""
    flag = "true" if use_chunking else "false"
    mode_str = "Chunked (908 items)" if use_chunking else "Full Doc (248 items)"
    print(f"\nRunning Ingestion Pipeline: {mode_str}...")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, "-m", "src.ingestion", flag]
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        encoding="utf-8", 
        errors="replace", 
        env=env
    )

    if result.returncode != 0:
        print(f"[ERROR] Ingestion failed:\n{result.stderr}")
        sys.exit(1)
        
    print(f"[OK] Ingestion complete for {mode_str}.\n")


def benchmark_mode(label: str, ground_truth: list) -> list:
    """Evaluates Keyword, Vector, and Hybrid search strategies for a specific ingestion mode."""
    results = []
    print(f"Benchmarking Strategies [{label}]...")
    print("-" * 65)
    print(f"{'Experiment':<35} | {'MRR':<10} | {'Hit Rate':<10}")
    print("-" * 65)

    strategies = [
        (f"{label} - Keyword", lambda s, q: s.search_keyword(q, top_n=5)),
        (f"{label} - Vector", lambda s, q: s.search_vector(q, top_n=5)),
        (f"{label} - Hybrid (RRF k=50)", lambda s, q: s.search_hybrid(q, top_n=5, rrf_k=50)),
    ]

    for exp_name, search_fn in strategies:
        # Re-instantiate searcher for a fresh DB connection pool per experiment
        searcher = PGSearcher()
        mrr_scores, hit_scores = [], []

        for record in ground_truth:
            query = record["question"]
            retrieved = search_fn(searcher, query)

            mrr_scores.append(calculate_mrr(retrieved, record))
            hit_scores.append(calculate_hit_rate(retrieved, record))

        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        avg_hit = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0

        results.append({
            "Experiment": exp_name,
            "MRR": round(avg_mrr, 4),
            "Hit Rate": round(avg_hit, 4),
        })

        print(f"{exp_name:<35} | {avg_mrr:.4f}     | {avg_hit:.4f}")

    print("-" * 65)
    return results


def run_full_ablation():
    """Runs end-to-end ablation across Full Doc and Chunked representations."""
    print("Loading ground truth dataset...")
    ground_truth = load_ground_truth("data/ground_truth.csv")
    print(f"[OK] Loaded {len(ground_truth)} records.")

    all_results = []

    # 1. Benchmark Full-Doc (248 items)
    run_ingestion(use_chunking=False)
    all_results.extend(benchmark_mode("Full Doc", ground_truth))

    # 2. Benchmark Chunked (908 items)
    run_ingestion(use_chunking=True)
    all_results.extend(benchmark_mode("Chunked", ground_truth))

    # 3. Save complete comparative matrix
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("data/evaluation_results.csv", index=False)
    print("\n[OK] Complete evaluation matrix saved to data/evaluation_results.csv\n")


if __name__ == "__main__":
    run_full_ablation()