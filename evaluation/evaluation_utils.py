import pandas as pd
from typing import List, Dict, Any


def load_ground_truth(file_path: str = "data/ground_truth.csv") -> List[Dict[str, Any]]:
    """Loads ground truth dataset from CSV."""
    df = pd.read_csv(file_path).fillna("")
    return df.to_dict(orient="records")


def clean_title(title_str: Any) -> str:
    """Normalizes titles by stripping whitespace and lowering case."""
    if not title_str:
        return ""
    return str(title_str).strip().lower()


def calculate_mrr(results: List[Dict[str, Any]], record: Dict[str, Any]) -> float:
    """Calculates MRR matching on title."""
    target_title = clean_title(record.get("title") or record.get("filename"))
    
    for rank, res in enumerate(results, start=1):
        retrieved_title = clean_title(res.get("title"))
        if target_title and retrieved_title and (target_title == retrieved_title or target_title in retrieved_title):
            return 1.0 / rank
    return 0.0


def calculate_hit_rate(results: List[Dict[str, Any]], record: Dict[str, Any]) -> float:
    """Calculates Hit Rate matching on title."""
    target_title = clean_title(record.get("title") or record.get("filename"))
    
    for res in results:
        retrieved_title = clean_title(res.get("title"))
        if target_title and retrieved_title and (target_title == retrieved_title or target_title in retrieved_title):
            return 1.0
    return 0.0


# Keep dummy wrapper to avoid __init__.py import errors
def normalize_doc_id(val: Any) -> str:
    return str(val)