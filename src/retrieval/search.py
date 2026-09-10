import os
import re
from typing import Any, Dict, List
import psycopg
from pgvector.psycopg import register_vector
from src.embedder import Embedder

DB_DSN = os.getenv("DB_DSN", "postgresql://app_user:app_password@localhost:5432/rag_db")

# Extended noise/stop word filter for clean OR query generation
STOP_WORDS = {
    "what", "which", "how", "where", "who", "when", "why", "are", "is", "was",
    "were", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with",
    "on", "at", "by", "from", "using", "paper", "papers", "study", "studies",
    "approach", "approaches", "method", "methods", "show", "used", "based",
    "results", "proposed", "performance", "analysis", "system", "model"
}


class PGSearcher:
    def __init__(self, model_path: str = "models/Xenova/all-MiniLM-L6-v2", rrf_k: int = 50):
        self.rrf_k = rrf_k
        self.embedder = Embedder(path=model_path)

    def _prepare_or_tsquery(self, query: str) -> str:
        """
        Converts natural language queries into an OR-based tsquery string
        after removing common prompt noise.
        """
        cleaned = re.sub(r"[^\w\s]", " ", query.lower())
        tokens = [
            token.strip() 
            for token in cleaned.split() 
            if len(token.strip()) > 2 and token.strip() not in STOP_WORDS
        ]
        if not tokens:
            tokens = [t.strip() for t in query.split() if len(t.strip()) > 1]
        return " | ".join(tokens)

    def search_keyword(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword search using weighted Full-Text Search.
        Uses normalization flag 32 in ts_rank_cd to divide rank by document length,
        stopping longer documents from skewing relevance scores.
        """
        or_query = self._prepare_or_tsquery(query)
        sql = """
            SELECT chunk_id, doc_id, title, abstract, text,
                   ts_rank_cd(fts, to_tsquery('english', %s), 32) AS score
            FROM document_chunks
            WHERE fts @@ to_tsquery('english', %s)
            ORDER BY score DESC LIMIT %s;
        """
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, (or_query, or_query, top_n))
                    return [self._format_row(r) for r in cur.fetchall()]
                except Exception:
                    # Fallback for complex characters
                    cur.execute(
                        "SELECT chunk_id, doc_id, title, abstract, text, "
                        "ts_rank_cd(fts, plainto_tsquery('english', %s), 32) AS score "
                        "FROM document_chunks WHERE fts @@ plainto_tsquery('english', %s) "
                        "ORDER BY score DESC LIMIT %s;",
                        (query, query, top_n)
                    )
                    return [self._format_row(r) for r in cur.fetchall()]

    def search_vector(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Dense Vector Search using PGVector Cosine Distance."""
        q_emb = self.embedder.encode(query, normalize=True).tolist()
        sql = """
            SELECT chunk_id, doc_id, title, abstract, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector ASC LIMIT %s;
        """
        with psycopg.connect(DB_DSN) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, (q_emb, q_emb, top_n))
                return [self._format_row(r) for r in cur.fetchall()]

    def search_hybrid(self, query: str, top_n: int = 5, rrf_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Hybrid Search combining Vector + Weighted FTS using Reciprocal Rank Fusion (RRF).
        Performs candidate collection up to rank 60 for better merge depth.
        """
        k_val = rrf_k if rrf_k is not None else self.rrf_k
        q_emb = self.embedder.encode(query, normalize=True).tolist()
        or_query = self._prepare_or_tsquery(query)

        sql = """
        WITH vector_search AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector ASC) AS rank
            FROM document_chunks ORDER BY embedding <=> %s::vector ASC LIMIT 60
        ),
        fts_search AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts, to_tsquery('english', %s), 32) DESC) AS rank
            FROM document_chunks WHERE fts @@ to_tsquery('english', %s) LIMIT 60
        )
        SELECT c.chunk_id, c.doc_id, c.title, c.abstract, c.text,
               COALESCE(1.0 / (%s + v.rank), 0.0) + COALESCE(1.0 / (%s + f.rank), 0.0) AS score
        FROM document_chunks c
        LEFT JOIN vector_search v ON c.chunk_id = v.chunk_id
        LEFT JOIN fts_search f ON c.chunk_id = f.chunk_id
        WHERE v.chunk_id IS NOT NULL OR f.chunk_id IS NOT NULL
        ORDER BY score DESC LIMIT %s;
        """
        with psycopg.connect(DB_DSN) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                try:
                    cur.execute(sql, (q_emb, q_emb, or_query, or_query, k_val, k_val, top_n))
                    return [self._format_row(r) for r in cur.fetchall()]
                except Exception:
                    fallback_sql = sql.replace("to_tsquery", "plainto_tsquery")
                    cur.execute(fallback_sql, (q_emb, q_emb, query, query, k_val, k_val, top_n))
                    return [self._format_row(r) for r in cur.fetchall()]

    def _format_row(self, row: tuple) -> Dict[str, Any]:
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "title": row[2],
            "abstract": row[3],
            "text": row[4],
            "score": float(row[5]),
        }


_default_searcher = None

def hybrid_search(query: str, top_n: int = 5, rrf_k: int = 50) -> List[Dict[str, Any]]:
    global _default_searcher
    if _default_searcher is None:
        _default_searcher = PGSearcher()
    return _default_searcher.search_hybrid(query=query, top_n=top_n, rrf_k=rrf_k)