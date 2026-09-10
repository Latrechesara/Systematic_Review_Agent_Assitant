import sys
from src.embedder import Embedder
from src.ingestion.chunk import prepare_dataset
from src.ingestion.index import (
    build_vector_embeddings, 
    save_to_pgvector, 
    clear_document_chunks
)

def main():
    # Toggle chunking via command-line argument: python -m src.ingestion true (or false)
    use_chunking = True
    if len(sys.argv) > 1:
        use_chunking = sys.argv[1].lower() == "true"

    mode_label = "CHUNKED (Sliding Window)" if use_chunking else "FULL ABSTRACT (No Chunking)"
    print(f"\n⚙️ Starting Ingestion Pipeline: {mode_label}")

    print("🧹 Step 1: Clearing previous database records...")
    clear_document_chunks()

    print("📦 Step 2: Loading dataset...")
    chunks = prepare_dataset(
        csv_path="data/lung_cancer_multimodal_papers.csv",
        use_chunking=use_chunking
    )
    
    print(f"🧠 Step 3: Generating embeddings for {len(chunks)} items using local ONNX Embedder...")
    embedder = Embedder()
    embeddings = build_vector_embeddings(chunks, embedder, batch_size=32)
    
    print("💾 Step 4: Persisting records to PGVector...")
    save_to_pgvector(chunks, embeddings)
    print("🚀 Ingestion Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()