from dotenv import load_dotenv
from openai import OpenAI
from ingestion import load_cancer_data, build_index
from rag_helper import LiteratureRAGBase

# Load environment variables from .env file
load_dotenv()

# Now OpenAI client will automatically detect OPENAI_API_KEY
client = OpenAI()

# 1. Load CSV data and initialize minsearch index
print("Loading CSV and building minsearch index...")
documents = load_cancer_data("data/lung_cancer_multimodal_papers.csv")
index = build_index(documents)
print(f"Indexed {len(documents)} papers successfully.")

# 2. Instantiate RAG assistant
assistant = LiteratureRAGBase(index=index, llm_client=client)

# 3. Query the assistant
query = "What deep learning approaches combine CT imaging with histopathology whole slide images for NSCLC prognosis?"
response = assistant.rag(query)

print("\n================ ASSISTANT ANSWER ================\n")
print(response["answer"])

print("\n================ RETRIEVED SOURCES ================\n")
for doc in response["retrieved_docs"]:
    print(f"- {doc['title']} ({doc['year']}) | Modalities: {doc['modalities']}")