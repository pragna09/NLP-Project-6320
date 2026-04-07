# src/rag.py
import json


def load_knowledge_base(filepath: str = "data/science_facts.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["knowledge_base"]


def retrieve_facts(problem: str, knowledge_base: list, top_k: int = 3) -> list:
    problem_words = set(problem.lower().split())
    scored = []
    for fact in knowledge_base:
        keywords = set(fact["keywords"])
        score = len(problem_words & keywords)
        if score > 0:
            scored.append({
                "fact":     fact["fact"],
                "category": fact["category"],
                "source":   fact["source"],
                "score":    score
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_facts_for_prompt(facts: list) -> str:
    if not facts:
        return "No relevant facts found."
    out = "Relevant Food Science Facts:\n"
    for i, f in enumerate(facts, 1):
        out += f"{i}. {f['fact']} (Source: {f['source']})\n"
    return out


def test_retrieval():
    print("Testing RAG retrieval...")
    kb = load_knowledge_base()
    print(f"Loaded {len(kb)} facts from knowledge base")

    problem = "My hollandaise sauce is broken and grainy"
    print(f"\nProblem: {problem}")

    facts = retrieve_facts(problem, kb, top_k=3)
    print(f"\n{format_facts_for_prompt(facts)}")
    print("Retrieval successful!")


if __name__ == "__main__":
    test_retrieval()