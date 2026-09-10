import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def sliding_window(seq: str, size: int, step: int) -> List[Dict[str, Any]]:
    """Creates overlapping text chunks using a sliding window."""
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    results = []
    for i in range(0, n, step):
        batch = seq[i : i + size]
        results.append({"start": i, "content": batch})
        if i + size >= n:
            break
    return results


def prepare_dataset(
    csv_path: str | Path = "data/lung_cancer_multimodal_papers.csv",
    use_chunking: bool = False,
    chunk_size: int = 1000,
    chunk_step: int = 500,
) -> List[Dict[str, Any]]:
    """Loads CSV papers and processes records for PGVector persistence."""
    df = pd.read_csv(csv_path).fillna("")
    raw_documents = []

    # 1. Parse raw records
    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        modalities = str(row.get("modalities_found_auto", "")).strip()
        combined_text = f"Title: {title}\nAbstract: {abstract}\nModalities: {modalities}"

        # Safe int conversion for year
        raw_year = row.get("year")
        year_val = None
        if pd.notna(raw_year) and str(raw_year).replace('.', '', 1).isdigit():
            year_val = int(float(raw_year))

        doc = {
            "doc_id": str(idx),
            "title": title,
            "abstract": abstract,
            "text": combined_text,
            "venue": str(row.get("venue", "")).strip(),
            "year": year_val,
            "authors": str(row.get("authors", "")).strip(),
            "modalities": modalities,
            "multimodal_score": str(row.get("multimodal_score", "")),
            "priority": str(row.get("priority", "")),
            "doi": str(row.get("doi", "")).strip(),
            "url": str(row.get("url", "")).strip(),
        }
        raw_documents.append(doc)

    # Strategy 1: Full Abstract (No Chunking)
    if not use_chunking:
        for doc in raw_documents:
            doc["chunk_id"] = f"doc_{doc['doc_id']}_full"
        return raw_documents

    # Strategy 2: Sliding Window Chunking
    chunked_records = []
    for doc in raw_documents:
        doc_copy = doc.copy()
        text_content = doc_copy.pop("text")
        windows = sliding_window(text_content, size=chunk_size, step=chunk_step)

        for ordinal, win in enumerate(windows):
            chunk = doc_copy.copy()
            chunk["chunk_id"] = f"doc_{doc['doc_id']}_c{ordinal}"
            chunk["text"] = win["content"]
            chunked_records.append(chunk)

    return chunked_records