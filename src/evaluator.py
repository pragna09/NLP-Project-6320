# src/evaluator.py
import json
import os
from datetime import datetime


def load_test_cases(filepath: str = "data/testcases.json") -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def score_cause_accuracy(predicted: str, ground_truth: str) -> float:
    stop = {"the","a","an","is","it","in","of","to","and","or","was"}
    p = set(predicted.lower().split()) - stop
    g = set(ground_truth.lower().split()) - stop
    if not g:
        return 0.0
    return min(len(p & g) / len(g), 1.0)


def score_solution_appropriateness(predicted: str, ground_truth: str) -> float:
    stop = {"the","a","an","is","it","in","of","to","and","or"}
    p = set(predicted.lower().split()) - stop
    g = set(ground_truth.lower().split()) - stop
    if not g:
        return 0.0
    return min(len(p & g) / len(g), 1.0)


def score_scientific_accuracy(response: str, keywords: list) -> float:
    if not response or not keywords:
        return 0.0
    r = response.lower()
    return min(sum(1 for k in keywords if k.lower() in r) / len(keywords), 1.0)


def calculate_overall_score(cause: float, solution: float, science: float) -> float:
    return round(cause * 0.40 + solution * 0.40 + science * 0.20, 4)


def parse_llm_response(response: str) -> dict:
    parsed  = {"cause": "", "solution": "", "explanation": ""}
    current = None
    for line in response.strip().split("\n"):
        ll = line.lower().strip()
        if ll.startswith("cause:"):
            current = "cause"
            parsed["cause"] = line.split(":", 1)[-1].strip()
        elif ll.startswith("solution:"):
            current = "solution"
            parsed["solution"] = line.split(":", 1)[-1].strip()
        elif ll.startswith("explanation:"):
            current = "explanation"
            parsed["explanation"] = line.split(":", 1)[-1].strip()
        elif current and line.strip():
            parsed[current] += " " + line.strip()
    return parsed


def evaluate_single(test_case: dict, llm_response: str,
                    science_keywords: list = None) -> dict:
    parsed = parse_llm_response(llm_response)
    cause  = score_cause_accuracy(
        parsed["cause"], test_case.get("Cause", ""))
    sol    = score_solution_appropriateness(
        parsed["solution"], test_case.get("Solution", ""))
    sci    = score_scientific_accuracy(llm_response, science_keywords or [])
    return {
        "test_case_id":       test_case.get("id"),
        "problem":            test_case.get("Problem"),
        "recipe":             test_case.get("Recipe"),
        "category":           test_case.get("Category"),
        "difficulty":         test_case.get("Difficulty"),
        "predicted_cause":    parsed["cause"],
        "ground_truth_cause": test_case.get("Cause", ""),
        "predicted_solution": parsed["solution"],
        "scores": {
            "cause_accuracy":           cause,
            "solution_appropriateness": sol,
            "scientific_accuracy":      sci,
            "overall": calculate_overall_score(cause, sol, sci)
        }
    }


def evaluate_all(results: list) -> dict:
    if not results:
        return {}
    def avg(lst): return round(sum(lst) / len(lst), 4)
    return {
        "total_cases": len(results),
        "average_scores": {
            "cause_accuracy":           avg([r["scores"]["cause_accuracy"] for r in results]),
            "solution_appropriateness": avg([r["scores"]["solution_appropriateness"] for r in results]),
            "scientific_accuracy":      avg([r["scores"]["scientific_accuracy"] for r in results]),
            "overall":                  avg([r["scores"]["overall"] for r in results])
        }
    }


def save_results(results: list, condition: str, model: str) -> str:
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"results/{condition}_{model}_{ts}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({
            "condition": condition,
            "model":     model,
            "timestamp": ts,
            "summary":   evaluate_all(results),
            "results":   results
        }, f, indent=2)
    print(f"Saved: {fn}")
    return fn


def test_evaluator():
    print("Testing Evaluator...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")

    mock_response = """
    CAUSE: Butter was added too quickly preventing proper emulsification
    SOLUTION: Whisk a fresh egg yolk with hot water and slowly add broken sauce
    EXPLANATION: Lecithin in egg yolk acts as emulsifier to rebind fat and water phases
    """

    result = evaluate_single(
        test_case=test_cases[0],
        llm_response=mock_response,
        science_keywords=["lecithin", "emulsifier", "emulsification"]
    )
    print(f"\nTest Case: {result['problem']}")
    print(f"Scores: {result['scores']}")
    print("\nEvaluator test successful!")


if __name__ == "__main__":
    test_evaluator()