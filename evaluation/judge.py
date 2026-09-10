import os
from typing import Literal, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

openai_client = OpenAI(timeout=30.0)

# ============================================================================
# 1. PYDANTIC SCHEMAS (Structured Outputs)
# ============================================================================

class BasicRAGAnswerEvaluation(BaseModel):
    reasoning: str = Field(description="Detailed explanation of correctness based on retrieved context and ground truth.")
    verdict: Literal["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"] = Field(
        description="RELEVANT: complete and accurate. PARTLY_RELEVANT: partial information. NON_RELEVANT: inaccurate or off-topic."
    )

class AgentTrajectoryEvaluation(BaseModel):
    reasoning: str = Field(description="Reasoning on tool choice, loops, and overall goal fulfillment.")
    tool_selection: Literal["YES", "NO"] = Field(description="'YES' if valid tools with clean args were used.")
    trajectory_efficiency: Literal["EFFICIENT", "INEFFICIENT"] = Field(description="'EFFICIENT' if no redundant loops occurred.")
    goal_completion: Literal["SUCCESS", "PARTIAL", "FAIL"] = Field(description="Final request completion status.")

# ============================================================================
# 2. PROMPTS & INSTRUCTIONS
# ============================================================================

BASIC_RAG_INSTRUCTIONS = """You are an expert evaluator assessing a Retrieval-Augmented Generation (RAG) system.
Compare the model's generated answer against the ground truth reference answer and context."""

BASIC_RAG_PROMPT = """User Question: {question}
Retrieved Context: {retrieved_context}
Ground Truth Answer: {ground_truth_answer}
Generated Answer: {generated_answer}"""

AGENT_TRAJECTORY_INSTRUCTIONS = """You are an expert evaluator analyzing an Agentic RAG system execution trajectory.
Evaluate tool usage, execution efficiency, and overall goal completion."""

AGENT_TRAJECTORY_PROMPT = """User Goal / Question: {question}
Execution Trajectory / Tool Calls: {execution_trace}
Final Response Output: {final_answer}"""

# ============================================================================
# 3. EVALUATION RUNNERS (With Retries)
# ============================================================================

@retry(
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5)
)
def judge_basic_rag(question: str, context: str, gt_answer: str, gen_answer: str, model: str = "gpt-4o-mini") -> Tuple[BasicRAGAnswerEvaluation, Any]:
    prompt = BASIC_RAG_PROMPT.format(
        question=question,
        retrieved_context=context,
        ground_truth_answer=gt_answer,
        generated_answer=gen_answer,
    )
    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": BASIC_RAG_INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ],
        response_format=BasicRAGAnswerEvaluation,
        temperature=0.0
    )
    return completion.choices[0].message.parsed, completion.usage


def judge_record(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """Worker function for ThreadPoolExecutor with flexible field mapping."""
    question = rec.get("question", "")
    
    # Ground truth answer fallbacks (maps 'expected_answer' from your ground truth dataset)
    gt_answer = rec.get("expected_answer") or rec.get("ground_truth_answer") or rec.get("text") or ""
    
    # Model output fallbacks
    gen_answer = (
        rec.get("rag_answer") or 
        rec.get("generated_answer") or 
        rec.get("answer_llm") or 
        rec.get("answer") or 
        ""
    )
    
    # Context fallbacks
    context = rec.get("context", rec.get("retrieved_context", rec.get("text", "")))

    # Guard clause against missing model output columns
    if not gen_answer.strip():
        return {
            "doc_id": rec.get("doc_id", "N/A"),
            "question": question,
            "verdict": "NON_RELEVANT",
            "reasoning": "ERROR: Generated model answer field was empty or missing in evaluation input dataset.",
        }, None

    eval_result, usage = judge_basic_rag(
        question=question,
        context=context,
        gt_answer=gt_answer,
        gen_answer=gen_answer
    )
    
    return {
        "doc_id": rec.get("doc_id", "N/A"),
        "question": question,
        "verdict": eval_result.verdict,
        "reasoning": eval_result.reasoning,
    }, usage

# ============================================================================
# 4. PARALLEL BENCHMARK EXECUTION
# ============================================================================

def run_judge_benchmark(
    dataset_path: str = "data/evaluation_results.csv",
    output_path: str = "data/eval_judge_results.csv",
    max_workers: int = 6
):
    if not os.path.exists(dataset_path):
        print(f"[!] Input dataset missing at {dataset_path}. Run evaluation pipeline first.")
        return

    df_gt = pd.read_csv(dataset_path)
    records = df_gt.to_dict(orient="records")
    
    print(f"Starting evaluation of {len(records)} records using {max_workers} threads...")

    evaluations = []
    usages = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(judge_record, records))

    for eval_dict, usage in results:
        evaluations.append(eval_dict)
        usages.append(usage)

    df_res = pd.DataFrame(evaluations)
    df_res.to_csv(output_path, index=False)
    
    print("\nScore Distribution:")
    print(df_res["verdict"].value_counts(normalize=True))
    print(f"\n[OK] Completed! Saved evaluation results to {output_path}")

if __name__ == "__main__":
    run_judge_benchmark()