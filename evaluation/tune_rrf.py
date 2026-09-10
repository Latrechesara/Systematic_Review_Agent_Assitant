from typing import List, Dict, Any
from evaluation.evaluation_utils import (
    load_ground_truth,
    calculate_mrr,
    calculate_hit_rate,
)
from src.retrieval import hybrid_search


def evaluate_rrf_k(ground_truth: List[Dict[str, Any]], rrf_k: int) -> Dict[str, float]:
    mrr_scores = []
    hit_rate_scores = []

    for item in ground_truth:
        query = item["question"]
        results = hybrid_search(query=query, rrf_k=rrf_k)

        mrr_scores.append(calculate_mrr(results, item))
        hit_rate_scores.append(calculate_hit_rate(results, item))

    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    avg_hit_rate = sum(hit_rate_scores) / len(hit_rate_scores) if hit_rate_scores else 0.0

    return {"rrf_k": rrf_k, "mrr": avg_mrr, "hit_rate": avg_hit_rate}


def main():
    print("🎯 Loading ground truth evaluation dataset...")
    ground_truth = load_ground_truth("data/ground_truth.csv")
    print(f"✓ Loaded {len(ground_truth)} ground truth records.\n")

    k_candidates = [1, 50, 100, 200]
    results = []

    print("📊 Evaluating Hybrid Search RRF Constant (k)")
    print(f"{'rrf_k':<10} | {'MRR':<10} | {'Hit Rate':<10}")
    print("-" * 36)

    for k in k_candidates:
        metrics = evaluate_rrf_k(ground_truth, rrf_k=k)
        results.append(metrics)
        print(f"{metrics['rrf_k']:<10} | {metrics['mrr']:.4f}     | {metrics['hit_rate']:.4f}")

    best_result = max(results, key=lambda x: (x["mrr"], -x["rrf_k"]))
    print("-" * 36)
    print(f"\n🏆 Optimal k value: {best_result['rrf_k']} (MRR: {best_result['mrr']:.4f})")


if __name__ == "__main__":
    main()