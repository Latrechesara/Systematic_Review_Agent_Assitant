import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.search import PGSearcher
from src.retrieval.tools import RETRIEVAL_TOOL, execute_tool_call

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

INSTRUCTIONS = """
You are an expert AI medical research assistant specialized in systematic literature reviews for multimodal lung cancer applications.
Your job is to answer research questions accurately using ONLY retrieved study abstracts.

Rules:
1. Ground every claim strictly in the provided paper context.
2. Format document IDs as 'doc_111' using the exact doc_id provided.
3. If the context does not contain enough evidence to answer, state: "I don't know based on the provided literature."
"""

RAG_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Evidence-based synthesis grounded strictly in retrieved abstracts.",
                },
                "citations": {
                    "type": "array",
                    "description": "List of paper citations used in the response.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Document ID cited (e.g., doc_111)",
                            },
                            "key_takeaway": {
                                "type": "string",
                                "description": "Key finding extracted from this specific paper",
                            },
                        },
                        "required": ["doc_id", "key_takeaway"],
                        "additionalProperties": False,
                    },
                },
                "is_sufficient": {
                    "type": "boolean",
                    "description": "False if context lacked sufficient evidence to answer.",
                },
            },
            "required": ["answer", "citations", "is_sufficient"],
            "additionalProperties": False,
        },
    },
}


class RAGPipeline:
    def __init__(self, searcher: Optional[PGSearcher] = None, model_name: str = "gpt-4o-mini"):
        self.searcher = searcher or PGSearcher()
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _build_context_string(self, docs: List[Dict[str, Any]]) -> str:
        blocks = []
        for d in docs:
            doc_id = d.get("doc_id", d.get("id", "unknown"))
            title = d.get("title", "")
            abstract = d.get("text", d.get("abstract", ""))
            blocks.append(
                f"--- Document ID: doc_{doc_id} ---\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}"
            )
        return "\n\n".join(blocks)

    def answer_query_basic(self, query: str, top_n: int = 5, search_mode: str = "hybrid") -> Dict[str, Any]:
        if search_mode == "keyword":
            docs = self.searcher.search_keyword(query, top_n=top_n)
        elif search_mode == "vector":
            docs = self.searcher.search_vector(query, top_n=top_n)
        else:
            docs = self.searcher.search_hybrid(query, top_n=top_n)

        if not docs:
            return {"answer": "No documents found.", "citations": [], "is_sufficient": False, "context": []}

        context_str = self._build_context_string(docs)
        prompt = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            response_format=RAG_RESPONSE_SCHEMA,
            temperature=0.1,
        )

        res = json.loads(response.choices[0].message.content)
        res["context"] = docs
        return res

    def answer_query_agentic(self, query: str, top_n: int = 5, max_turns: int = 4) -> Dict[str, Any]:
        messages = [
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": query},
        ]
        
        accumulated_docs = []

        for _ in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[RETRIEVAL_TOOL],
                temperature=0.1,
            )

            message = response.choices[0].message
            messages.append(message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    retrieved = self.searcher.search_hybrid(query=args.get("query", query), top_n=top_n)
                    accumulated_docs.extend(retrieved)
                    
                    tool_msg = execute_tool_call(tool_call, self.searcher, top_n=top_n)
                    messages.append(tool_msg)
            else:
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=RAG_RESPONSE_SCHEMA,
                    temperature=0.1,
                )
                res = json.loads(final_response.choices[0].message.content)
                res["context"] = accumulated_docs
                return res

        return {
            "answer": "I don't know based on the provided literature.",
            "citations": [],
            "is_sufficient": False,
            "context": [],
        }