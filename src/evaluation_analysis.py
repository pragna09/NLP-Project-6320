# src/evaluation_analysis.py
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns         


# ─────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────

def load_all_results(results_dir: str = "results") -> list:
    """
    Loads all JSON result files from the results/ folder.
    """
    all_results = []
    files = glob.glob(f"{results_dir}/*.json")

    if not files:
        print("No result files found in results/ folder.")
        return []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results.append(data)
        print(f"Loaded: {filepath}")

    print(f"\nTotal files loaded: {len(all_results)}")
    return all_results


def results_to_dataframe(all_results: list) -> pd.DataFrame:
    """
    Converts all result files into a single flat DataFrame
    for easy analysis.
    """
    rows = []
    for result_file in all_results:
        condition = result_file.get("condition")
        model     = result_file.get("model")
        for r in result_file.get("results", []):
            rows.append({
                "condition":                condition,
                "model":                    model,
                "test_case_id":             r.get("test_case_id"),
                "problem":                  r.get("problem"),
                "recipe":                   r.get("recipe"),
                "category":                 r.get("category"),
                "difficulty":               r.get("difficulty"),
                "cause_accuracy":           r["scores"]["cause_accuracy"],
                "solution_appropriateness": r["scores"]["solution_appropriateness"],
                "scientific_accuracy":      r["scores"]["scientific_accuracy"],
                "overall":                  r["scores"]["overall"],
            })

    df = pd.DataFrame(rows)
    print(f"DataFrame created: {len(df)} rows")
    return df


# ─────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────

def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average scores grouped by condition.
    Shows how each condition performed overall.
    """
    return df.groupby("condition")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average scores grouped by model.
    Shows how each LLM performed overall.
    """
    return df.groupby("model")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_condition_and_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average scores grouped by condition AND model.
    The main comparison table for your paper.
    """
    return df.groupby(["condition", "model"])[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average scores grouped by difficulty level.
    Shows if harder problems score lower.
    """
    return df.groupby("difficulty")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average scores grouped by recipe category.
    Shows which categories are harder to diagnose.
    """
    return df.groupby("category")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


# ─────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────

def anova_by_condition(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA test across the 4 conditions.
    Tests if condition differences are statistically significant.

    Returns F-statistic and p-value.
    p < 0.05 means the differences are significant.
    """
    from scipy import stats

    groups = [
        df[df["condition"] == c]["overall"].values
        for c in df["condition"].unique()
    ]

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": round(f_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05
    }


def anova_by_model(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA test across the 4 models.
    Tests if model differences are statistically significant.
    """
    from scipy import stats

    groups = [
        df[df["model"] == m]["overall"].values
        for m in df["model"].unique()
    ]

    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": round(f_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05
    }


def hallucination_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates hallucination rate per condition.
    Low scientific_accuracy = likely hallucination.
    Threshold: scientific_accuracy < 0.2
    """
    df["hallucinated"] = df["scientific_accuracy"] < 0.2
    return df.groupby("condition")["hallucinated"].mean().round(4).reset_index()


# ─────────────────────────────────────────
# SAVE ANALYSIS
# ─────────────────────────────────────────

def save_analysis(df: pd.DataFrame, output_dir: str = "analysis"):
    """
    Saves all summary tables to the analysis/ folder as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each summary table
    tables = {
        "by_condition":           summarize_by_condition(df),
        "by_model":               summarize_by_model(df),
        "by_condition_and_model": summarize_by_condition_and_model(df),
        "by_difficulty":          summarize_by_difficulty(df),
        "by_category":            summarize_by_category(df),
    }

    for name, table in tables.items():
        path = f"{output_dir}/{name}.csv"
        table.to_csv(path)
        print(f"Saved: {path}")

    # Save full dataframe
    full_path = f"{output_dir}/full_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    # Save ANOVA results
    anova_results = {
        "anova_by_condition": anova_by_condition(df),
        "anova_by_model":     anova_by_model(df),
        "hallucination_rates": hallucination_rate(df).to_dict(orient="records")
    }

    anova_path = f"{output_dir}/anova_results.json"
    with open(anova_path, "w") as f:
        json.dump(anova_results, f, indent=2)
    print(f"Saved: {anova_path}")

    print(f"\nAll analysis saved to {output_dir}/")


# ─────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────

def print_report(df: pd.DataFrame):
    """
    Prints a full summary report to the terminal.
    """
    print("\n" + "=" * 60)
    print("R.E.C.I.P.E. EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal trials: {len(df)}")
    print(f"Conditions:   {df['condition'].nunique()}")
    print(f"Models:       {df['model'].nunique()}")
    print(f"Test cases:   {df['test_case_id'].nunique()}")

    print("\n--- By Condition ---")
    print(summarize_by_condition(df).to_string())

    print("\n--- By Model ---")
    print(summarize_by_model(df).to_string())

    print("\n--- By Condition and Model ---")
    print(summarize_by_condition_and_model(df).to_string())

    print("\n--- By Difficulty ---")
    print(summarize_by_difficulty(df).to_string())

    print("\n--- By Category ---")
    print(summarize_by_category(df).to_string())

    print("\n--- ANOVA by Condition ---")
    anova_c = anova_by_condition(df)
    print(f"  F-statistic: {anova_c['f_statistic']}")
    print(f"  p-value:     {anova_c['p_value']}")
    print(f"  Significant: {anova_c['significant']}")

    print("\n--- ANOVA by Model ---")
    anova_m = anova_by_model(df)
    print(f"  F-statistic: {anova_m['f_statistic']}")
    print(f"  p-value:     {anova_m['p_value']}")
    print(f"  Significant: {anova_m['significant']}")

    print("\n--- Hallucination Rates by Condition ---")
    print(hallucination_rate(df).to_string(index=False))

    print("\n" + "=" * 60)

# ─────────────────────────────────────────
# PLOTS 
# ─────────────────────────────────────────
  
def generate_plots(df: pd.DataFrame, output_dir: str = "analysis/plots"):
    """
    Creates visual charts to compare RAG vs. Baseline performance.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Overall Performance by Condition and Model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="condition", y="overall", hue="model", data=df, palette="magma")
    plt.title("R.E.C.I.P.E. Performance: Model & Condition Comparison", fontsize=15)
    plt.ylabel("Overall Score (Weighted)")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_results.png")
    plt.close()

    # 2. Metric Breakdown Heatmap
    metrics = ["cause_accuracy", "solution_appropriateness", "scientific_accuracy"]
    pivot_df = df.groupby("condition")[metrics].mean()

    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Score'})
    plt.title("Score Distribution Across Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_heatmap.png")
    plt.close()

    print(f"Plots generated in {output_dir}/")


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

def run_analysis():
    """
    Main function — loads all results, prints report,
    and saves analysis to analysis/ folder.
    """
    print("Loading results...")
    all_results = load_all_results()

    if not all_results:
        print("No results to analyze yet.")
        print("Run experiments first: python main.py --mode full")
        return

    df = results_to_dataframe(all_results)
    print_report(df)
    save_analysis(df)
    
    # Generate the visual plots
    print("Generating visualizations...")
    generate_plots(df)


if __name__ == "__main__":
    run_analysis()