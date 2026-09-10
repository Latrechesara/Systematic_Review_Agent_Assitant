from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATA_PATH = PROJECT_ROOT / "data" / "lung_cancer_multimodal_papers.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ground_truth.csv"


class QAItem(BaseModel):
    question: str = Field(
        description="A natural search query or question answered by the abstract."
    )
    expected_answer: str = Field(
        description="A concise, factually accurate answer derived directly from the paper abstract."
    )


class Questions(BaseModel):
    items: list[QAItem] = Field(
        description="5 distinct, high-quality search queries and their corresponding answers derived from the abstract."
    )


DATA_GEN_INSTRUCTIONS = """
You emulate a medical researcher conducting a systematic review on Multimodal AI for Lung Cancer.
You are given a research paper title and abstract.
Formulate 5 distinct question/answer pairs that this researcher might query and verify against this paper.

Rules:
- The abstract MUST contain the clear answer to each question.
- Cover PICO criteria (Population, Intervention, Comparator, Outcome), fusion modalities (e.g., CT, WSI, PET, Omics), and performance metrics where applicable.
- Make the questions complete and realistic search queries.
- Ensure the expected_answer is a factual summary extracted directly from the text without verbatim copying.
""".strip()


def generate_ground_truth():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find input file at {DATA_PATH}")

    client = OpenAI()
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["abstract"]).reset_index(drop=True)

    print(f"Loaded {len(df)} papers with valid abstracts from {DATA_PATH}.")

    ground_truth_records = []

    for idx, row in df.iterrows():
        title = row["title"]
        abstract = row["abstract"]
        
        # Ensure a clean doc_id is assigned for evaluation matching
        doc_id = row.get("doc_id", row.get("id", f"paper_{idx+1}"))

        user_prompt = f"{DATA_GEN_INSTRUCTIONS}\n\nTitle: {title}\nAbstract: {abstract}"

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_prompt}],
                response_format=Questions,
            )

            generated_items = response.choices[0].message.parsed.items

            for item in generated_items:
                ground_truth_records.append(
                    {
                        "doc_id": str(doc_id),
                        "question": item.question,
                        "expected_answer": item.expected_answer,
                        "title": title,
                        "modalities": row.get("modalities_found_auto", ""),
                    }
                )

            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                print(f"Processed {idx + 1}/{len(df)} papers...")

        except Exception as e:
            print(f"Error generating for index {idx} ({title[:30]}...): {e}")
            continue

    gt_df = pd.DataFrame(ground_truth_records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Successfully saved {len(gt_df)} ground truth records to {OUTPUT_PATH}!"
    )


if __name__ == "__main__":
    generate_ground_truth()