import json
from typing import Any, Dict


# Schema definition using OpenAI Function Calling standard
RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_medical_documents",
        "description": "Searches medical literature abstracts based on a search query using hybrid search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant medical abstracts.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def execute_tool_call(tool_call: Any, searcher: Any, top_n: int = 5) -> Dict[str, Any]:
    """Executes search function calls requested by the agent and formats tool output."""
    args = json.loads(tool_call.function.arguments)

    if tool_call.function.name == "retrieve_medical_documents":
        docs = searcher.search_hybrid(query=args["query"], top_n=top_n)

        # Build context string with explicit doc_id boundaries
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

        context_str = "\n\n".join(blocks) if blocks else "No relevant documents found."

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": context_str,
        }

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": "Unknown tool requested.",
    }