from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET

import pandas as pd

# ============================
# Config
# ============================

START_YEAR = 2020
END_YEAR = 2026

PUBMED_RETMAX = 200
DBLP_RETMAX = 300
REQUEST_SLEEP_SECONDS = 0.4
MAX_RETRIES = 3

EMAIL_FOR_PUBMED = "latreche.sara93@gmail.com"
TOOL_NAME = "strict_multimodal_lung_cancer_review"

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Updated clinical terms for Lung Cancer
LUNG_CANCER_TERMS = [
    "lung cancer",
    "lung neoplasm",
    "non-small cell lung cancer",
    "nsclc",
    "small cell lung cancer",
    "sclc",
    "lung adenocarcinoma",
    "squamous cell lung carcinoma",
    "pulmonary nodule",
    "lung carcinoma",
]

FUSION_TERMS = [
    "multimodal",
    "multi-modal",
    "fusion",
    "cross-modal",
    "cross modal",
    "multimodal learning",
    "multimodal deep learning",
    "joint learning",
    "joint representation",
    "cooperative learning",
    "cross-attention",
    "cross attention",
]

# Modality dictionary updated for Oncology/Lung Cancer
MODALITY_TERMS = {
    "ct": ["computed tomography", "ct", "low-dose ct", "ldct", "radiomics"],
    "pet": ["positron emission tomography", "pet", "pet-ct", "pet/ct"],
    "xray": ["chest x-ray", "cxr", "radiograph"],
    "mri": ["mri", "magnetic resonance"],
    "pathology": ["histopathology", "whole slide imaging", "wsi", "pathological", "biopsy"],
    "omics": ["genomic", "transcriptomic", "rna-seq", "mutation", "multi-omics", "egfr", "alk", "proteomics"],
    "clinical": ["clinical data", "tabular", "electronic health record", "ehr", "clinical notes"],
    "liquid_biopsy": ["liquid biopsy", "ctdna", "circulating tumor dna"],
}

AI_TERMS = [
    "deep learning",
    "neural network",
    "transformer",
    "representation learning",
    "self-supervised learning",
    "machine learning",
    "cnn",
    "vit",
    "vision transformer",
    "gnn",
]

EXCLUDE_HINT_TERMS = [
    "review",
    "systematic review",
    "meta-analysis",
    "protocol",
    "preprint",
]

# ============================
# Data model
# ============================

@dataclass
class Paper:
    source: str
    title: str
    authors: str
    year: Optional[int]
    venue: str
    abstract: str
    doi: str
    url: str
    query_used: str

# ============================
# Helpers
# ============================

def http_get(url: str, headers: Optional[Dict[str, str]] = None, sleep_s: float = REQUEST_SLEEP_SECONDS) -> str:
    last_error = None
    merged_headers = {"User-Agent": f"{TOOL_NAME}/1.0 ({EMAIL_FOR_PUBMED})"}
    if headers:
        merged_headers.update(headers)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers=merged_headers)
            with urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            time.sleep(sleep_s)
            return data
        except (HTTPError, URLError, TimeoutError) as e:
            last_error = e
            wait_time = attempt * 2
            print(f"[warn] Request failed ({attempt}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {last_error}")

def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def normalize_title(title: str) -> str:
    title = normalize_text(title).lower()
    title = re.sub(r"[^a-z0-9 ]", "", title)
    return title

def normalize_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return doi

def safe_year_from_text(text: Optional[str]) -> Optional[int]:
    if not text: return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None

def find_keywords(text: str, keywords: List[str]) -> List[str]:
    text = text.lower()
    return [kw for kw in keywords if kw.lower() in text]

def detect_modalities(text: str) -> List[str]:
    text = text.lower()
    found = []
    for modality, keywords in MODALITY_TERMS.items():
        if any(k in text for k in keywords):
            found.append(modality)
    return found

def build_strict_multimodal_query() -> str:
    cancer = " OR ".join(f'"{t}"' for t in LUNG_CANCER_TERMS)
    fusion = " OR ".join(f'"{t}"' for t in FUSION_TERMS)
    ai = " OR ".join(f'"{t}"' for t in AI_TERMS)
    return f"({cancer}) AND ({fusion}) AND ({ai})"

def looks_like_review_or_protocol_or_preprint(p: Paper) -> bool:
    text = f"{p.title} {p.abstract} {p.venue}".lower()
    return any(term in text for term in EXCLUDE_HINT_TERMS)

def is_true_multimodal_candidate(p: Paper) -> bool:
    text = f"{p.title} {p.abstract}".lower()

    fusion_found = find_keywords(text, FUSION_TERMS)
    modalities_found = detect_modalities(text)
    ai_found = find_keywords(text, AI_TERMS)
    cancer_found = find_keywords(text, LUNG_CANCER_TERMS)

    if not (fusion_found and ai_found and cancer_found):
        return False

    modset = set(modalities_found)
    if len(modset) < 2:
        return False

    # Filter weak sets for Lung Cancer research
    weak_sets = [{"xray", "clinical"}, {"xray", "clinical", "ehr"}]
    if modset in [set(s) for s in weak_sets]:
        return False

    return True

# ============================
# PubMed & DBLP Search Logic
# ============================

def pubmed_search(term: str, start_year: int, end_year: int, retmax: int = PUBMED_RETMAX) -> List[str]:
    query = f'({term}) AND ("{start_year}"[Date - Publication] : "{end_year}"[Date - Publication])'
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    all_pmids: List[str] = []
    retstart = 0

    while True:
        url = (f"{base_url}?db=pubmed&retmode=json&retmax={retmax}&retstart={retstart}"
               f"&term={quote(query)}&tool={quote(TOOL_NAME)}&email={quote(EMAIL_FOR_PUBMED)}")
        raw = http_get(url)
        data = json.loads(raw)
        result = data.get("esearchresult", {})
        idlist = result.get("idlist", [])
        count = int(result.get("count", "0"))
        all_pmids.extend(idlist)
        retstart += retmax
        if len(all_pmids) >= count or not idlist: break
    return all_pmids

def pubmed_fetch_details(pmids: List[str], query_used: str, batch_size: int = 100) -> List[Paper]:
    if not pmids: return []
    papers: List[Paper] = []
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        url = f"{base_url}?db=pubmed&id={','.join(batch)}&retmode=xml&tool={quote(TOOL_NAME)}&email={quote(EMAIL_FOR_PUBMED)}"
        raw = http_get(url)
        root = ET.fromstring(raw)

        for article in root.findall(".//PubmedArticle"):
            title = normalize_text("".join(article.find(".//ArticleTitle").itertext())) if article.find(".//ArticleTitle") is not None else ""
            
            abstract_parts = []
            for abst in article.findall(".//Abstract/AbstractText"):
                label = abst.attrib.get("Label", "").strip()
                text = normalize_text("".join(abst.itertext()))
                abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts)

            authors = []
            for a in article.findall(".//Author"):
                last, fore, coll = a.findtext("LastName", ""), a.findtext("ForeName", ""), a.findtext("CollectiveName", "")
                authors.append(coll if coll else f"{fore} {last}".strip())

            venue = normalize_text(article.findtext(".//Journal/Title", ""))
            
            year = None
            for ys in [article.findtext(".//PubDate/Year"), article.findtext(".//ArticleDate/Year")]:
                year = safe_year_from_text(ys)
                if year: break

            doi, pmid = "", ""
            for aid in article.findall(".//ArticleId"):
                it, txt = aid.attrib.get("IdType"), (aid.text or "").strip()
                if it == "doi": doi = normalize_doi(txt)
                elif it == "pubmed": pmid = txt

            p = Paper("PubMed", title, "; ".join(authors), year, venue, abstract, doi, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "", query_used)
            if is_true_multimodal_candidate(p): papers.append(p)
    return papers

def dblp_search(term: str, start_year: int, end_year: int, max_results: int = DBLP_RETMAX) -> List[Paper]:
    url = f"https://dblp.org/search/publ/api?q={quote(term)}&h={max_results}&format=json"
    raw = http_get(url)
    hits = json.loads(raw).get("result", {}).get("hits", {}).get("hit", [])
    papers = []

    for hit in hits:
        info = hit.get("info", {})
        year = int(info.get("year")) if info.get("year") else None
        if year and not (start_year <= year <= end_year): continue

        p = Paper(
            source="DBLP",
            title=normalize_text(info.get("title", "")),
            authors="; ".join([a.get("text", "") if isinstance(a, dict) else str(a) for a in (info.get("authors", {}).get("author", []) if isinstance(info.get("authors", {}).get("author", []), list) else [info.get("authors", {}).get("author", {})])]),
            year=year,
            venue=normalize_text(info.get("venue", "")),
            abstract="",
            doi=normalize_doi(info.get("doi", "")),
            url=normalize_text(info.get("url", "")),
            query_used=term,
        )
        if is_true_multimodal_candidate(p): papers.append(p)
    return papers

# ============================
# Deduplication & Scoring
# ============================

def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    seen_doi, seen_title, deduped = set(), set(), []
    for p in papers:
        dk, tk = normalize_doi(p.doi), normalize_title(p.title)
        if (dk and dk in seen_doi) or (tk and tk in seen_title): continue
        if dk: seen_doi.add(dk)
        if tk: seen_title.add(tk)
        deduped.append(p)
    return deduped

def multimodal_score(p: Paper) -> tuple[int, List[str], List[str]]:
    text = f"{p.title} {p.abstract}".lower()
    fusion_found = find_keywords(text, FUSION_TERMS)
    modalities_found = detect_modalities(text)
    
    score = 5 if fusion_found else 0
    modset = set(modalities_found)
    if len(modset) >= 2: score += 4
    if len(modset) >= 3: score += 2

    # High-impact Lung Cancer modality pairs
    strong_pairs = [{"ct", "omics"}, {"ct", "pathology"}, {"pet", "ct"}, {"ct", "ehr"}]
    for pair in strong_pairs:
        if pair.issubset(modset): score += 3
            
    return score, fusion_found, modalities_found

def screen_papers(papers: List[Paper]) -> List[Dict[str, str]]:
    screened = []
    for p in papers:
        score, f_found, m_found = multimodal_score(p)
        priority = "high" if score >= 10 else "medium" if score >= 7 else "low"
        
        row = asdict(p)
        row.update({
            "fusion_keywords_found": "; ".join(f_found),
            "modalities_found_auto": "; ".join(m_found),
            "multimodal_score": score,
            "priority": priority,
            "screen_flag": "manual_check" if looks_like_review_or_protocol_or_preprint(p) else "include_candidate",
            "exclusion_hint": "Possible review/preprint" if looks_like_review_or_protocol_or_preprint(p) else "",
            "include_decision": "",
            "exclude_reason_final": "",
            "notes": ""
        })
        screened.append(row)
    return screened

# ============================
# Save functions
# ============================

def save_csv(rows: List[Dict[str, str]], filename: Path) -> None:
    if not rows:
        print("[warn] No rows to save to CSV.")
        return
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def save_excel(rows: List[Dict[str, str]], filename: Path) -> None:
    if not rows:
        print("[warn] No rows to save to Excel.")
        return
    df = pd.DataFrame(rows)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    df["priority_order"] = df["priority"].map(priority_order)
    df = df.sort_values(by=["priority_order", "multimodal_score", "year"], ascending=[True, False, False]).drop(columns=["priority_order"])
    df.to_excel(filename, index=False)

# ============================
# Main
# ============================

def main() -> None:
    query = build_strict_multimodal_query()
    print(f"Searching Lung Cancer Multimodal papers...\nQuery: {query}\n")

    print("Searching PubMed...")
    pmids = pubmed_search(query, START_YEAR, END_YEAR)
    pubmed_papers = pubmed_fetch_details(pmids, query_used=query)
    print(f"PubMed papers found: {len(pubmed_papers)}")

    print("Searching DBLP...")
    dblp_papers = dblp_search(query, START_YEAR, END_YEAR)
    print(f"DBLP papers found: {len(dblp_papers)}")

    all_papers = deduplicate_papers(pubmed_papers + dblp_papers)
    print(f"Combined and Deduplicated: {len(all_papers)}")

    screened = screen_papers(all_papers)

    csv_file = OUTPUT_DIR / "lung_cancer_multimodal_papers.csv"
    xlsx_file = OUTPUT_DIR / "lung_cancer_multimodal_papers.xlsx"

    save_csv(screened, csv_file)
    save_excel(screened, xlsx_file)

    print(f"\nFinished! Processed {len(screened)} papers.")
    print(f"Saved CSV:   {csv_file}")
    print(f"Saved Excel: {xlsx_file}")

if __name__ == "__main__":
    main()