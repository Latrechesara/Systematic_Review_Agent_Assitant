INSTRUCTIONS = """
You are an expert AI medical research assistant specialized in systematic literature reviews for multimodal lung cancer applications.
Your job is to answer research questions accurately using ONLY the provided study abstracts.

Rules:
1. Ground every claim strictly in the provided paper context.
2. If the context does not contain enough evidence to answer, state: "I don't know based on the provided literature."
3. Always cite relevant paper titles, publication years, and modality combinations in your synthesis.
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT FROM LUNG CANCER LITERATURE:
{context}
""".strip()


class LiteratureRAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-4o-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str, num_results: int = 5, filter_dict: dict = None) -> list[dict]:
        """Performs TF-IDF field-weighted keyword search via minsearch."""
        boost_dict = {
            "title": 3.0,
            "modalities": 2.0,
            "abstract": 1.0,
            "venue": 0.5
        }
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict or {}
        )

    def build_context(self, search_results: list[dict]) -> str:
        """Formats retrieved papers into structured text blocks for the prompt."""
        lines = []
        for i, doc in enumerate(search_results, 1):
            lines.append(f"[{i}] Title: {doc['title']} ({doc['year']})")
            lines.append(f"    Modalities: {doc['modalities']} | Venue: {doc['venue']}")
            lines.append(f"    Abstract: {doc['abstract']}")
            lines.append(f"    DOI/URL: {doc['doi'] if doc['doi'] else doc['url']}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: list[dict]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            temperature=0.1
        )
        return response.choices[0].message.content

    def rag(self, query: str, filter_dict: dict = None) -> dict:
        search_results = self.search(query, filter_dict=filter_dict)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return {
            "answer": answer,
            "retrieved_docs": search_results,
            "prompt": prompt
        }