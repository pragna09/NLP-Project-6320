# main.py
import argparse
from src.experiment_runner import run_all_experiments, test_runner


def parse_args():
    parser = argparse.ArgumentParser(description="R.E.C.I.P.E.")
    parser.add_argument(
        "--mode", default="test",
        choices=["test", "full", "single", "analyze"],
        help="test=2 cases, full=all cases, single=interactive, analyze=run analysis"
    )
    parser.add_argument(
        "--cases", type=int, default=None,
        help="Number of test cases to run"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between API calls in seconds"
    )
    return parser.parse_args()


def run_single_interactive():
    from src.models import call_llama_big, call_llama_fast
    from src.rag import load_knowledge_base, load_recipes, retrieve_facts, format_facts_for_prompt
    from src.knowledge_graphs import (
        load_substitutions,
        build_fkg, build_skg,
        query_fkg, find_critical_steps, map_problem_to_step
    )
    from src.experiment_runner import build_full_system_prompt

    print("\nLoading data...")
    kb      = load_knowledge_base()
    recipes = load_recipes()
    fkg     = build_fkg(recipes)
    skg     = build_skg(load_substitutions())
    print("Ready!\n")

    while True:
        print("-" * 50)
        problem = input("Describe your cooking problem (or 'quit'):\n> ")
        if problem.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        recipe   = input("What recipe are you making?\n> ")
        steps    = query_fkg(fkg, recipe)
        critical = find_critical_steps(fkg, recipe)
        mapped   = map_problem_to_step(fkg, recipe, problem)
        facts    = format_facts_for_prompt(
            retrieve_facts(problem, recipe, kb, recipes, top_k=3))
        prompt   = build_full_system_prompt(
            problem, recipe, steps, critical, mapped, facts)

        print("\nAnalyzing with both models...")

        response_llama_big  = call_llama_big(prompt)
        response_llama_fast = call_llama_fast(prompt)

        print("\n" + "=" * 50)
        print("LLaMA 3.3 70B DIAGNOSIS:")
        print("=" * 50)
        print(response_llama_big)

        print("\n" + "=" * 50)
        print("LLaMA 3.1 8B DIAGNOSIS:")
        print("=" * 50)
        print(response_llama_fast)
        print("=" * 50 + "\n")


def main():
    args = parse_args()
    print("=" * 50)
    print("R.E.C.I.P.E.")
    print("=" * 50)
    print(f"Mode: {args.mode}\n")

    if args.mode == "test":
        test_runner()
    elif args.mode == "full":
        run_all_experiments(max_cases=args.cases, delay=args.delay)
    elif args.mode == "single":
        run_single_interactive()
    elif args.mode == "analyze":
        from src.evaluation_analysis import run_analysis
        run_analysis()


if __name__ == "__main__":
    main()