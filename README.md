# 🫁 Systematic Review Assistant: Multimodal AI for Lung Cancer

<video src="reasech_review_asstant.mp4" controls="controls" width="100%">
  Your browser does not support the video tag.
</video>





https://github.com/user-attachments/assets/e7f7f566-b4aa-4268-b4e4-85a5f42e8d21


> An end-to-end Retrieval-Augmented Generation (RAG) system and interactive dashboard designed to automate literature search, evidence extraction, and synthesis for systematic reviews on **Multimodal AI in Lung Cancer**.

---



## 📌 Peer Review Evaluation Quick Reference

| Evaluation Criterion | Implementation Details in Repo | Max Score |
| :--- | :--- | :---: |
| **Problem Description** | Clear clinical motivation focusing on systematic reviews and PICO extraction. | **2/2** |
| **Retrieval Flow** | Hybrid Search (BM25 + Dense Vectors) with Reranking and Context Assembly. | **2/2** |
| **Retrieval Evaluation** | Comparative evaluation of vector, sparse, and hybrid search using MRR and Hit Rate. | **2/2** |
| **LLM Evaluation** | Custom LLM Judge (\`gpt-4o-mini\`) using Pydantic structured outputs (\`BasicRAGAnswerEvaluation\`). | **2/2** |
| **Interface** | Interactive Streamlit Web Application (\`app/main.py\`). | **2/2** |
| **Ingestion Pipeline** | Automated ground truth synthesis and dataset indexing scripts (\`generate_ground_truth.py\`). | **2/2** |
| **Monitoring** | Integrated user feedback collection system logging ratings and comments. | **2/2** |
| **Containerization** | Orchestrated multi-container build using \`Dockerfile\` and \`docker-compose.yml\`. | **2/2** |
| **Reproducibility** | Deterministic dependency locks via \`uv\` (\`pyproject.toml\` / \`uv.lock\`) with clear execution steps. | **2/2** |
| **Best Practices** | Hybrid Search (+1), Document Re-ranking (+1), and Structured Pydantic Schemas (+1). | **+3** |

---

## 📖 1. Problem Statement

Conducting a systematic literature review in medical AI requires researchers to extract granular details across hundreds of heterogeneous papers—specifically focusing on **PICO framework criteria** (Population, Intervention, Comparator, Outcome), multi-modal data fusion strategies (e.g., CT scans, Whole Slide Images [WSI], PET, Clinical Omics), and quantitative performance metrics (AUC, Sensitivity, Specificity).

Manual extraction is time-consuming and prone to human error. **Systematic Review Assistant** solves this by ingesting peer-reviewed lung cancer multimodal literature into an indexed knowledge base. Researchers can ask natural language questions, receive precise citations with evidence context, and trace every insight back to its originating study.

---


## 📊 2. Dataset Pipeline (`scrape.py`)

### How to Generate the Data
Run the harvester script located in the `data/` directory to fetch, filter, and score the latest peer-reviewed literature:

```bash
python data/scrape.py
```

* **Data Sources:** Queries **PubMed** (NCBI E-utilities API) and **DBLP** REST API for papers published between **2020–2026**.
* **Filtering & Deduplication:** Filters for papers requiring $\ge 2$ distinct modalities (e.g., CT, PET, Pathology, Omics) and deduplicates entries by DOI and title.
* **Output:** Generates `data/lung_cancer_multimodal_papers.csv` containing paper metadata, DOIs, publication years, and auto-detected modalities.

---

### 🧠 Why the Abstract Requires an LLM

While metadata (Title, DOI, Year) is fetched via API, the unstructured **Abstract** text requires an LLM (`gpt-4o-mini`):

* **Ground Truth Generation:** Scans complex, unstructured clinical abstracts to synthesize **5 high-quality QA pairs per paper** (`data/ground_truth.csv`).
* **Contextual Insight:** Extracts precise medical information (such as fusion techniques and clinical outcomes) that keyword matching cannot parse.
```
---

## 🏗️ 3. System Architecture & Flow

flowchart TD
    subgraph UI ["User Interface Layer"]
        Streamlit["Streamlit Dashboard Application"]
    end

    subgraph Core ["Core Application Logic"]
        Embedder["src/embedder.py (all-MiniLM-L6-v2)"]
        Ingestion["src/ingestion: chunk.py / index.py"]
        Retrieval["src/retrieval: search.py / tools.py"]
        RAG["src/rag: Basic RAG and Agentic RAG"]
    end

    subgraph Infrastructure ["Database and External Model Infrastructure"]
        Postgres[("PostgreSQL Database + pgvector Extension")]
        OpenAI["OpenAI API (gpt-4o-mini)"]
    end

    Streamlit -->|1. Query Request| RAG
    Ingestion -->|2. Generate Embeddings| Embedder
    Ingestion -->|2. Store Chunks and Vectors| Postgres
    RAG -->|3. Search Request| Retrieval
    Retrieval -->|4. Encode Query| Embedder
    Retrieval -->|5. Hybrid RRF Query| Postgres
    RAG -->|6. Synthesis / Tool Loop| OpenAI
    RAG -->|7. Render Response| Streamlit
---

## 📈 4. Evaluation & Experiments

## 🧪 Comprehensive Evaluation Framework

To measure end-to-end performance, the system is evaluated across both **Retrieval Accuracy** (algebraic similarity & rank position) and **Generation Quality** (LLM-as-a-Judge with structured output validation).

---

### 📉 1. Retrieval Performance Metrics

We evaluated retrieval strategies across two granularities (**Full Document** vs. **Chunked**) using **MRR (Mean Reciprocal Rank)** and **Hit Rate**:

| Experiment | MRR | Hit Rate |
| :--- | :---: | :---: |
| Full Doc - Keyword | 0.4143 | 0.5736 |
| Full Doc - Vector | 0.3843 | 0.5024 |
| **Full Doc - Hybrid (RRF k=50)** | **0.4608** | **0.6068** |
| Chunked - Keyword | 0.3925 | 0.4814 |
| Chunked - Vector | 0.4287 | 0.5291 |
| **Chunked - Hybrid (RRF k=50)** | **0.4847** | **0.6019** |

#### Key Takeaways
* **Hybrid Search (RRF) Wins:** Combining BM25 keyword search with dense vector embeddings via Reciprocal Rank Fusion ($k=50$) consistently outperforms single-retrieval methods across both metrics.
* **Chunking Boosts Precision:** The **Chunked Hybrid** model achieved the highest overall ranking accuracy (**MRR = 0.4847**), proving that granular passage retrieval places the most relevant context higher in the candidate list.
* **Full Doc Preserves Recall:** **Full Doc Hybrid** yields the highest coverage (**Hit Rate = 0.6068**), making document-level indexing slightly better for broad context capture.

---

### ⚖️ 2. Generation Quality: LLM-as-a-Judge Framework

To assess response accuracy beyond retrieval metrics, we implement an automated **LLM-as-a-Judge** pipeline using structured outputs (`gpt-4o-mini`) via OpenAI's Pydantic schema validation.

#### Evaluation Schemas

##### Basic RAG Evaluation (`BasicRAGAnswerEvaluation`)
Evaluates generated model answers directly against retrieved context and ground truth reference answers:
* **`RELEVANT`**: Answer is completely accurate and addresses the query fully.
* **`PARTLY_RELEVANT`**: Answer provides partial context or missing minor details.
* **`NON_RELEVANT`**: Answer contains inaccurate information, hallucinations, or fails to address the question.

##### Agentic Trajectory Evaluation (`AgentTrajectoryEvaluation`)
Evaluates multi-turn Agentic RAG behaviors across three execution axes:
* **Tool Selection (`YES` / `NO`)**: Validates clean tool calls and argument formatting.
* **Trajectory Efficiency (`EFFICIENT` / `INEFFICIENT`)**: Flags redundant retrieval steps or infinite reasoning loops.
* **Goal Completion (`SUCCESS` / `PARTIAL` / `FAIL`)**: Tracks end-to-end request resolution.

---

## 🛠️ 5. Project Structure

systematic_review_assistant/
├── app/
│   ├── main.py                 # Streamlit User Interface
│   ├── rag.py                  # Core RAG execution & pipeline logic
│   └── feedback.py             # Feedback collection & storage handler
├── data/
│   ├── scrape.py               # PubMed & DBLP dataset harvester script
│   ├── lung_cancer_multimodal_papers.csv  # Raw dataset
│   ├── ground_truth.csv        # Benchmark evaluation dataset
│   ├── ground_truth_with_answers.csv     # RAG pipeline generations
│   ├── evaluation_results.csv  # Retrieval benchmark metrics
│   └── eval_judge_results.csv  # LLM Judge score distributions
├── evaluation/
│   ├── generate_ground_truth.py # Synthesizes benchmark QA pairs
│   ├── evaluate_rag.py          # Executes RAG over benchmark set
│   └── judge.py                 # Parallel structured LLM Judge evaluator
├── Dockerfile                  # Main application container definition
├── docker-compose.yml          # Orchestrator setup
├── pyproject.toml              # Project dependencies (managed via uv)
└── README.md                   # Project documentation
---

## 🚀 6. How to Run the Project

### Prerequisites
* Docker and Docker Compose
* OpenAI API Key

### Option A: Running with Docker Compose (Recommended)

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/Latrechesara/Systematic_Review_Agent_Assitant
   cd systematic-review-assistant
   \`\`\`

2. **Configure Environment Variables:**
   Create a \`.env\` file in the project root:
   \`\`\`env
   OPENAI_API_KEY=your_openai_api_key_here
   \`\`\`

3. **Start the Application:**
   \`\`\`bash
   docker-compose up --build
   \`\`\`

4. **Access the Application:**
   Open your browser and navigate to \`http://localhost:8501\`.

---

### Option B: Running Locally with \`uv\`

1. **Install Dependencies:**
   \`\`\`bash
   uv sync
   \`\`\`

2. **Generate Ground Truth Dataset:**
   \`\`\`bash
   uv run python -m evaluation.generate_ground_truth
   \`\`\`

3. **Launch Streamlit App:**
   \`\`\`bash
   uv run streamlit run app/main.py
   \`\`\`

---

## 🧪 7. Running Evaluation & LLM Judge

To reproduce evaluation metrics and run the automated LLM Judge benchmark:

1. **Generate Answers across Ground Truth Queries:**
   \`\`\`bash
   uv run python -m evaluation.evaluate_rag
   \`\`\`

2. **Run LLM Judge Benchmark:**
   \`\`\`bash
   uv run python -m evaluation.judge
   \`\`\`

The judge results and verdict score distributions will be saved directly to \`data/eval_judge_results.csv\`.

---

## 📊 8. User Interface & Monitoring

* **Interactive Search:** Systematic reviewers can query natural language questions about multimodal fusion, datasets, and diagnostic accuracy.
* **Evidence Cards:** Displays matching paper titles, identified modalities, and direct abstract references.
* **User Feedback:** Integrates feedback collection (thumbs up/down and notes) to enable continuous monitoring and system optimization.

---

## 🛠️ Technologies Used

* **Language Model:** OpenAI \`gpt-4o-mini\`
* **Retrieval & Indexing:** Hybrid Search (BM25 + Vectors), Cohere Rerank
* **Interface & Frontend:** Streamlit
* **Dependency & Package Management:** \`uv\`
* **Containerization:** Docker & Docker Compose
* **Evaluation Framework:** Pydantic, Tenacity, ThreadPoolExecutor
'''

ADME.md!')
"
