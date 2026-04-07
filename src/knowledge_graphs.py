# src/knowledge_graphs.py
import json
import networkx as nx


def load_recipes(filepath: str = "data/recipes.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["recipes"]


def load_substitutions(filepath: str = "data/substitution_graph.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fkg(recipes: list) -> nx.DiGraph:
    fkg = nx.DiGraph()
    for recipe in recipes:
        name  = recipe["name"]
        steps = recipe.get("steps", [])
        for i, step in enumerate(steps):
            nid = f"{name}_step_{step['step']}"
            fkg.add_node(nid,
                recipe=name,
                step_number=step["step"],
                action=step["action"],
                critical_point=step.get("critical_point"))
            if i > 0:
                pid = f"{name}_step_{steps[i-1]['step']}"
                fkg.add_edge(pid, nid)
    return fkg


def query_fkg(fkg: nx.DiGraph, recipe_name: str) -> list:
    steps = [
        {
            "step":           d["step_number"],
            "action":         d["action"],
            "critical_point": d["critical_point"]
        }
        for _, d in fkg.nodes(data=True)
        if d.get("recipe") == recipe_name
    ]
    return sorted(steps, key=lambda x: x["step"])


def find_critical_steps(fkg: nx.DiGraph, recipe_name: str) -> list:
    return [s for s in query_fkg(fkg, recipe_name) if s["critical_point"]]


def map_problem_to_step(fkg: nx.DiGraph, recipe_name: str, problem: str) -> dict:
    words = set(problem.lower().split())
    best, best_score = None, 0
    for _, d in fkg.nodes(data=True):
        if d.get("recipe") == recipe_name and d.get("critical_point"):
            score = len(words & set(d["critical_point"].lower().split()))
            if score > best_score:
                best_score = score
                best = {
                    "step":           d["step_number"],
                    "action":         d["action"],
                    "critical_point": d["critical_point"]
                }
    return best


def build_skg(substitutions: list) -> nx.Graph:
    skg = nx.Graph()
    for entry in substitutions:
        for ing, data in entry.get("ingredients", {}).items():
            skg.add_node(ing,
                functions=data["properties"].get("functions", []))
            for sub in data.get("substitutes", []):
                skg.add_node(sub["name"])
                skg.add_edge(ing, sub["name"],
                    ratio=sub["ratio"],
                    why=sub["why"],
                    best_for=sub["best_for"],
                    avoid_for=sub["avoid_for"])
    return skg


def find_substitutes(skg: nx.Graph, ingredient: str) -> list:
    if ingredient not in skg:
        return []
    return [
        {"substitute": n, **skg.get_edge_data(ingredient, n)}
        for n in skg.neighbors(ingredient)
    ]


def format_substitutes_for_prompt(ingredient: str, substitutes: list) -> str:
    if not substitutes:
        return f"No substitutes found for {ingredient}."
    out = f"Substitutes for {ingredient}:\n"
    for i, sub in enumerate(substitutes, 1):
        out += f"{i}. {sub['substitute']} - Ratio: {sub['ratio']}\n"
        out += f"   Why: {sub['why']}\n"
    return out


def test_graphs():
    print("Testing Knowledge Graphs...")
    recipes       = load_recipes()
    substitutions = load_substitutions()
    print(f"Loaded {len(recipes)} recipes")
    print(f"Loaded {len(substitutions)} substitution entries")

    fkg = build_fkg(recipes)
    skg = build_skg(substitutions)
    print(f"\nFKG: {fkg.number_of_nodes()} nodes, {fkg.number_of_edges()} edges")
    print(f"SKG: {skg.number_of_nodes()} nodes, {skg.number_of_edges()} edges")

    recipe_name = "Hollandaise Sauce"
    steps = query_fkg(fkg, recipe_name)
    print(f"\nSteps for {recipe_name}:")
    for s in steps:
        print(f"  Step {s['step']}: {s['action']}")

    problem = "My hollandaise sauce is thin and won't thicken"
    mapped  = map_problem_to_step(fkg, recipe_name, problem)
    print(f"\nProblem: {problem}")
    print(f"Mapped to: {mapped}")

    subs = find_substitutes(skg, "butter")
    print(f"\nSubstitutes for butter:")
    print(format_substitutes_for_prompt("butter", subs))
    print("Knowledge Graphs test successful!")


if __name__ == "__main__":
    test_graphs()