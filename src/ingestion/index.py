import os
from typing import List, Dict, Any
import psycopg
from pgvector.psycopg import register_vector
from src.embedder import Embedder

DB_DSN = os.getenv("DB_DSN", "postgresql://app_user:app_password@localhost:5432/rag_db")


def clear_document_chunks() -> None:
    """Empties the document_chunks table before loading a new ingestion mode."""
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
            conn.commit()
    print("🧹 Cleared existing records from document_chunks.")


def init_db() -> None:
    """Initializes pgvector extension, database schema, and weighted full-text search indexes."""
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            # Enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)

            # Create document chunks table with 384d vector column
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(100) UNIQUE,
                    doc_id VARCHAR(100),
                    title TEXT,
                    abstract TEXT,
                    modalities TEXT,
                    venue TEXT,
                    year INT,
                    priority VARCHAR(50),
                    text TEXT,
                    embedding vector(384)
                );
            """)

            # Drop old fts column to safely re-apply new weighted tsvector definition
            cur.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS fts CASCADE;")

            # Add weighted generated tsvector column (A = Title, B = Modalities, C = Abstract & Text)
            cur.execute("""
                ALTER TABLE document_chunks 
                ADD COLUMN fts tsvector 
                GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(modalities, '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(abstract, '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(text, '')), 'C')
                ) STORED;
            """)

            # Create GIN index for weighted full-text querying
            cur.execute("""
                CREATE INDEX IF NOT EXISTS fts_weighted_idx 
                ON document_chunks USING GIN (fts);
            """)

            # Create HNSW index for fast vector lookup
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_hnsw_idx 
                ON document_chunks USING hnsw (embedding vector_cosine_ops);
            """)

            conn.commit()
    print("✓ PGVector schema, weighted FTS, and HNSW indexes successfully initialized.")


def build_vector_embeddings(
    chunks: List[Dict[str, Any]], 
    embedder: Embedder, 
    batch_size: int = 32
) -> List[List[float]]:
    """Generates normalized 384d embeddings using the local ONNX Embedder in batches."""
    texts = [c["text"] for c in chunks]
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.encode_batch(batch, normalize=True)
        embeddings_list.extend(batch_embeddings.tolist())

    return embeddings_list


def save_to_pgvector(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """Inserts or updates document chunks and corresponding local embeddings in PostgreSQL."""
    init_db()

    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                chunk_id = chunk.get(
                    "chunk_id", 
                    f"doc_{chunk.get('doc_id')}_c{chunk.get('chunk_index', 0)}"
                )

                cur.execute("""
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, title, abstract, modalities, 
                        venue, year, priority, text, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        abstract = EXCLUDED.abstract,
                        modalities = EXCLUDED.modalities,
                        venue = EXCLUDED.venue,
                        year = EXCLUDED.year,
                        priority = EXCLUDED.priority,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding;
                """, (
                    chunk_id,
                    str(chunk.get("doc_id", "")),
                    chunk.get("title", ""),
                    chunk.get("abstract", ""),
                    str(chunk.get("modalities", "")),
                    chunk.get("venue", ""),
                    chunk.get("year", None),
                    chunk.get("priority", ""),
                    chunk.get("text", ""),
                    emb
                ))
            conn.commit()
    print(f"✓ Successfully persisted {len(chunks)} chunks into PGVector database.")