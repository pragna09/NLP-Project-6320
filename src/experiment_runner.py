# src/experiment_runner.py
import time
from src.models import call_llama_big, call_llama_fast
from src.rag import load_knowledge_base, load_recipes, retrieve_facts, format_facts_for_prompt
from src.knowledge_graphs import (
    load_substitutions,
    build_fkg, build_skg,
    query_fkg, find_critical_steps,
    map_problem_to_step, find_substitutes,
    format_substitutes_for_prompt
)
from src.evaluator import load_test_cases, evaluate_single, evaluate_all, save_results


# ─────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────

def build_baseline_prompt(problem: str, recipe: str, steps: list) -> str:
    s = "\n".join(f"Step {s['step']}: {s['action']}" for s in steps)
    return (
        f"You are a culinary expert.\n"
        f"Problem: {problem}\n"
        f"Recipe: {recipe}\n\n"
        f"Recipe Steps:\n{s}\n\n"
        "Please diagnose the problem and provide a solution.\n"
        "Format your response exactly as:\n"
        "CAUSE: [what went wrong]\n"
        "SOLUTION: [how to fix it]\n"
        "EXPLANATION: [the science behind it]"
    )


def build_cot_prompt(problem: str, recipe: str, steps: list) -> str:
    s = "\n".join(f"Step {s['step']}: {s['action']}" for s in steps)
    return (
        f"You are a culinary expert.\n"
        f"Problem: {problem}\n"
        f"Recipe: {recipe}\n\n"
        f"Recipe Steps:\n{s}\n\n"
        "Think through this step by step:\n"
        "1. Which step most likely caused this problem?\n"
        "2. What is the scientific reason this step failed?\n"
        "3. What is the best solution to fix this?\n"
        "4. How can this be prevented in the future?\n\n"
        "Format your final response exactly as:\n"
        "CAUSE: [what went wrong]\n"
        "SOLUTION: [how to fix it]\n"
        "EXPLANATION: [the science behind it]"
    )


def build_kg_augmented_prompt(problem: str, recipe: str, steps: list,
                               critical: list, mapped: dict,
                               retrieved_facts: str) -> str:
    s = "\n".join(f"Step {s['step']}: {s['action']}" for s in steps)
    c = "\n".join(
        f"Step {s['step']}: {s['critical_point']}"
        for s in critical if s["critical_point"]
    )
    m = (f"\nMost Likely Failed Step:\n"
         f"Step {mapped['step']}: {mapped['action']}\n"
         f"Critical Point: {mapped['critical_point']}\n") if mapped else ""
    return (
        f"You are a culinary expert with deep food science knowledge.\n"
        f"Problem: {problem}\n"
        f"Recipe: {recipe}\n\n"
        f"Recipe Steps:\n{s}\n\n"
        f"Critical Control Points:\n{c}\n"
        f"{m}\n"
        f"{retrieved_facts}\n\n"
        "Using the above context, think through this step by step:\n"
        "1. Which critical control point was violated?\n"
        "2. What is the chemical or physical reason?\n"
        "3. What is the scientifically grounded solution?\n\n"
        "CAUSE: [what went wrong]\n"
        "SOLUTION: [how to fix it]\n"
        "EXPLANATION: [the science behind it]"
    )


def build_full_system_prompt(problem: str, recipe: str, steps: list,
                              critical: list, mapped: dict,
                              retrieved_facts: str,
                              substitutes_text: str = "") -> str:
    s = "\n".join(f"Step {s['step']}: {s['action']}" for s in steps)
    c = "\n".join(
        f"Step {s['step']}: {s['critical_point']}"
        for s in critical if s["critical_point"]
    )
    m = (f"\nMost Likely Failed Step:\n"
         f"Step {mapped['step']}: {mapped['action']}\n"
         f"Critical Point: {mapped['critical_point']}\n") if mapped else ""
    subs = f"\nIngredient Substitution Options:\n{substitutes_text}" \
           if substitutes_text else ""
    return (
        f"You are a culinary expert and food scientist.\n"
        f"Problem: {problem}\n"
        f"Recipe: {recipe}\n\n"
        f"Recipe Steps:\n{s}\n\n"
        f"Critical Control Points:\n{c}\n"
        f"{m}\n"
        f"{retrieved_facts}\n"
        f"{subs}\n\n"
        "Using ALL the above context, think through this carefully:\n"
        "1. Map the symptom to the exact failed step\n"
        "2. Retrieve the relevant food science principle\n"
        "3. Reason from symptom to root cause to solution\n"
        "4. Verify your solution does not cause a secondary problem\n\n"
        "CAUSE: [what went wrong - be specific about the science]\n"
        "SOLUTION: [step by step fix with measurements where relevant]\n"
        "EXPLANATION: [the food science behind the problem and solution]"
    )


# ─────────────────────────────────────────
# RUN SINGLE CASE
# ─────────────────────────────────────────

def run_single_case(test_case, condition, model_fn,
                    fkg, skg, knowledge_base, recipes) -> dict:
    """Run a single test case through one experimental condition"""

    problem     = test_case.get("Problem", "")
    recipe_name = test_case.get("Recipe", "")

    # Get recipe information from FKG
    steps    = query_fkg(fkg, recipe_name)
    critical = find_critical_steps(fkg, recipe_name)
    mapped   = map_problem_to_step(fkg, recipe_name, problem)

    # Retrieve relevant facts using RAG (with recipe awareness)
    facts     = retrieve_facts(problem, recipe_name, knowledge_base, recipes, top_k=3)
    retrieved = format_facts_for_prompt(facts)

    # ── Extract science keywords from retrieved facts ──────────
    science_keywords = []
    for fact in facts:
        kb_fact = next(
            (f for f in knowledge_base if f["fact"] == fact["fact"]),
            None
        )
        if kb_fact:
            science_keywords.extend(kb_fact.get("keywords", []))
    science_keywords = list(set(science_keywords))
    # ──────────────────────────────────────────────────────────

    # Build prompt based on condition
    if condition == "baseline":
        prompt = build_baseline_prompt(problem, recipe_name, steps)
    elif condition == "cot_only":
        prompt = build_cot_prompt(problem, recipe_name, steps)
    elif condition == "kg_augmented":
        prompt = build_kg_augmented_prompt(
            problem, recipe_name, steps, critical, mapped, retrieved)
    elif condition == "full_system":
        subs_text = ""
        for ing in ["butter", "egg", "milk"]:
            subs = find_substitutes(skg, ing)
            if subs:
                subs_text += format_substitutes_for_prompt(ing, subs)
        prompt = build_full_system_prompt(
            problem, recipe_name, steps, critical, mapped, retrieved, subs_text)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Query model and evaluate
    response = model_fn(prompt)
    result   = evaluate_single(
        test_case,
        response,
        science_keywords=science_keywords    # ← now passed correctly
    )
    result["condition"]    = condition
    result["raw_response"] = response
    return result


# ─────────────────────────────────────────
# RUN ALL EXPERIMENTS
# ─────────────────────────────────────────

def run_all_experiments(max_cases: int = None, delay: float = 2.0):
    """Run complete experiment suite"""

    print("=" * 50)
    print("Starting R.E.C.I.P.E. Experiments")
    print("=" * 50)

    # Load all data
    test_cases     = load_test_cases()
    knowledge_base = load_knowledge_base()
    recipes        = load_recipes()
    fkg            = build_fkg(recipes)
    skg            = build_skg(load_substitutions())

    print(f"Loaded {len(test_cases)} test cases")
    print(f"Loaded {len(knowledge_base)} science facts")
    print(f"Loaded {len(recipes)} recipes")
    print(f"FKG: {fkg.number_of_nodes()} nodes")
    print(f"SKG: {skg.number_of_nodes()} nodes")

    if max_cases:
        test_cases = test_cases[:max_cases]
        print(f"\nLimited to {max_cases} test cases")

    # Define models and conditions
    models = {
        "LLaMA 3.3 70B": call_llama_big,
        "LLaMA 3.1 8B":  call_llama_fast
    }
    conditions = ["baseline", "cot_only", "kg_augmented", "full_system"]

    # Run experiments
    for model_name, model_fn in models.items():
        for condition in conditions:
            print(f"\n{'─' * 40}")
            print(f"Running: {condition} | {model_name}")
            print(f"{'─' * 40}")

            results = []
            for i, tc in enumerate(test_cases):
                print(f"  {i+1}/{len(test_cases)}: {tc.get('Problem','')[:50]}...")
                try:
                    result = run_single_case(
                        tc, condition, model_fn,
                        fkg, skg, knowledge_base, recipes)
                    results.append(result)
                    print(f"  Score: {result['scores']['overall']:.3f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                time.sleep(delay)

            # Save and summarize results
            if results:
                save_results(results, condition, model_name)
                summary = evaluate_all(results)
                print(f"\nSummary — {condition} | {model_name}:")
                print(f"  Overall:  {summary['average_scores']['overall']:.3f}")
                print(f"  Cause:    {summary['average_scores']['cause_accuracy']:.3f}")
                print(f"  Solution: {summary['average_scores']['solution_appropriateness']:.3f}")
                print(f"  Science:  {summary['average_scores']['scientific_accuracy']:.3f}")

    print("\n" + "=" * 50)
    print("All experiments complete!")
    print("Results saved to results/")
    print("=" * 50)


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

def test_runner():
    """Quick test with limited cases"""
    print("Running quick test with 2 cases...")
    run_all_experiments(max_cases=2, delay=1.0)
    print("Test complete!")


if __name__ == "__main__":
    test_runner()