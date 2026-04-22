# src/rag.py
import json


# ─────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────

def load_knowledge_base(filepath: str = "data/science_facts.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["knowledge_base"]


def load_recipes(filepath: str = "data/recipes.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)["recipes"]


# ─────────────────────────────────────────
# RECIPE LOOKUP
# ─────────────────────────────────────────

def get_recipe_by_name(recipe_name: str, recipes: list) -> dict:
    """
    Finds a recipe by name from the recipe database.
    Returns None if not found.
    """
    for recipe in recipes:
        if recipe["name"].lower() == recipe_name.lower():
            return recipe
    return None


def extract_recipe_context(recipe: dict) -> dict:
    """
    Extracts ingredients, critical points and common problems
    from a recipe — used to build a targeted search query.

    Returns a dict with:
    - ingredient_names: list of ingredient names
    - critical_keywords: keywords from critical points
    - problem_keywords: keywords from common problems
    """
    if not recipe:
        return {
            "ingredient_names": [],
            "critical_keywords": [],
            "problem_keywords": []
        }

    # Extract ingredient names
    ingredient_names = []
    for ing in recipe.get("ingredients", []):
        name = ing.get("name", "").lower()
        if name:
            # Split multi-word ingredients e.g. "egg yolks" → ["egg", "yolks"]
            ingredient_names.extend(name.split())

    # Extract keywords from critical points in steps
    critical_keywords = []
    for step in recipe.get("steps", []):
        cp = step.get("critical_point", "")
        if cp:
            critical_keywords.extend(cp.lower().split())

    # Extract keywords from common problems
    problem_keywords = []
    for problem in recipe.get("common_problems", []):
        problem_keywords.extend(problem.lower().split())

    # Clean stop words
    stop_words = {
        "the", "a", "an", "is", "it", "in", "of", "to",
        "and", "or", "was", "be", "with", "at", "by",
        "for", "on", "are", "as", "if", "too", "not"
    }
    ingredient_names  = [w for w in ingredient_names  if w not in stop_words]
    critical_keywords = [w for w in critical_keywords if w not in stop_words]
    problem_keywords  = [w for w in problem_keywords  if w not in stop_words]

    return {
        "ingredient_names":  list(set(ingredient_names)),
        "critical_keywords": list(set(critical_keywords)),
        "problem_keywords":  list(set(problem_keywords))
    }


# ─────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────

def retrieve_facts(
    problem: str,
    recipe_name: str,
    knowledge_base: list,
    recipes: list,
    top_k: int = 3
) -> list:
    """
    Retrieves the most relevant science facts for a given problem
    by first anchoring to the specific recipe's ingredients and
    critical points — not just the raw problem text.

    Scoring priority:
    1. Ingredient match  — fact keyword matches a recipe ingredient
    2. Critical point match — fact keyword matches a recipe critical point
    3. Problem text match — fact keyword matches the problem description
    """

    # Step 1: Find the recipe
    recipe = get_recipe_by_name(recipe_name, recipes)
    if not recipe:
        print(f"  Warning: Recipe '{recipe_name}' not found — falling back to problem text only")

    # Step 2: Extract recipe context
    context = extract_recipe_context(recipe)
    ingredient_names  = set(context["ingredient_names"])
    critical_keywords = set(context["critical_keywords"])
    problem_keywords  = set(problem.lower().split())

    # Step 3: Score each fact
    scored_facts = []
    for fact in knowledge_base:
        fact_keywords = set(kw.lower() for kw in fact.get("keywords", []))
        fact_text     = fact.get("fact", "").lower()
        fact_words    = set(fact_text.split())

        # Score 1: ingredient overlap (highest weight — recipe-specific)
        ingredient_score = len(fact_keywords & ingredient_names) * 3

        # Score 2: critical point overlap (medium weight)
        critical_score = len(fact_keywords & critical_keywords) * 2

        # Score 3: problem text overlap (lower weight — too generic)
        problem_score = len(fact_keywords & problem_keywords) * 1

        # Score 4: fact text itself contains ingredient names
        text_ingredient_score = len(fact_words & ingredient_names) * 2

        total_score = (
            ingredient_score +
            critical_score +
            problem_score +
            text_ingredient_score
        )

        if total_score > 0:
            scored_facts.append({
                "fact":               fact["fact"],
                "category":           fact["category"],
                "source":             fact["source"],
                "score":              total_score,
                "ingredient_matches": list(fact_keywords & ingredient_names),
                "critical_matches":   list(fact_keywords & critical_keywords),
            })

    # Step 4: Sort by score descending
    scored_facts.sort(key=lambda x: x["score"], reverse=True)

    return scored_facts[:top_k]


def format_facts_for_prompt(facts: list) -> str:
    """
    Formats retrieved facts into a string for the LLM prompt.
    """
    if not facts:
        return "No relevant science facts found."

    out = "Relevant Food Science Facts:\n"
    for i, f in enumerate(facts, 1):
        out += f"{i}. {f['fact']} (Source: {f['source']})\n"
        if f.get("ingredient_matches"):
            out += f"   Relevant ingredients: {', '.join(f['ingredient_matches'])}\n"
    return out


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

def test_retrieval():
    print("Testing RAG retrieval...\n")

    kb      = load_knowledge_base()
    recipes = load_recipes()
    print(f"Loaded {len(kb)} science facts")
    print(f"Loaded {len(recipes)} recipes")

    # Test case 1 — sauce emulsion
    problem     = "My hollandaise sauce is thin and won't thicken up"
    recipe_name = "Hollandaise Sauce"
    print(f"\nProblem:  {problem}")
    print(f"Recipe:   {recipe_name}")

    recipe  = get_recipe_by_name(recipe_name, recipes)
    context = extract_recipe_context(recipe)
    print(f"\nRecipe ingredients found: {context['ingredient_names']}")
    print(f"Critical keywords found:  {context['critical_keywords'][:10]}")

    facts = retrieve_facts(problem, recipe_name, kb, recipes, top_k=3)
    print(f"\n{format_facts_for_prompt(facts)}")

    # Test case 2 — baking
    problem     = "My cookies spread into one giant flat sheet"
    recipe_name = "Chocolate Chip Cookies"
    print(f"\nProblem:  {problem}")
    print(f"Recipe:   {recipe_name}")

    facts = retrieve_facts(problem, recipe_name, kb, recipes, top_k=3)
    print(f"\n{format_facts_for_prompt(facts)}")

    print("Retrieval test successful!")


if __name__ == "__main__":
    test_retrieval()