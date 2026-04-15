# src/evaluation_analysis.py
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# ─────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────

def load_all_results(results_dir: str = "results", latest_only: bool = False) -> list:
    """
    Loads all JSON result files from the results/ folder.
    If latest_only=True, only loads the most recent run.
    """
    all_results = []
    files = glob.glob(f"{results_dir}/*.json")

    if not files:
        print("No result files found in results/ folder.")
        return []

    if latest_only:
        groups = defaultdict(list)
        for f in files:
            key = "_".join(os.path.basename(f).split("_")[:-2])
            groups[key].append(f)
        files = [sorted(v)[-1] for v in groups.values()]
        print(f"Latest only mode — loading {len(files)} files")

    for filepath in sorted(files):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_results.append(data)
        print(f"Loaded: {filepath}")

    print(f"\nTotal files loaded: {len(all_results)}")
    return all_results


def results_to_dataframe(all_results: list) -> pd.DataFrame:
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
    return df.groupby("condition")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("model")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_condition_and_model(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["condition", "model"])[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("difficulty")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("category")[
        ["cause_accuracy", "solution_appropriateness",
         "scientific_accuracy", "overall"]
    ].mean().round(4)


# ─────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────

def anova_by_condition(df: pd.DataFrame) -> dict:
    from scipy import stats
    groups = [
        df[df["condition"] == c]["overall"].values
        for c in df["condition"].unique()
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": round(float(f_stat), 4),    # ← add float()
        "p_value":     round(float(p_value), 6),   # ← add float()
        "significant": bool(p_value < 0.05)        # ← add bool()
    }


def anova_by_model(df: pd.DataFrame) -> dict:
    from scipy import stats
    groups = [
        df[df["model"] == m]["overall"].values
        for m in df["model"].unique()
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    return {
        "f_statistic": round(float(f_stat), 4),    # ← add float()
        "p_value":     round(float(p_value), 6),   # ← add float()
        "significant": bool(p_value < 0.05)        # ← add bool()
    }


def hallucination_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hallucinated"] = df["scientific_accuracy"] < 0.2
    return df.groupby("condition")["hallucinated"].mean().round(4).reset_index()


# ─────────────────────────────────────────
# SAVE ANALYSIS
# ─────────────────────────────────────────

def save_analysis(df: pd.DataFrame, output_dir: str = "analysis"):
    os.makedirs(output_dir, exist_ok=True)

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

    full_path = f"{output_dir}/full_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

    anova_results = {
        "anova_by_condition":  anova_by_condition(df),
        "anova_by_model":      anova_by_model(df),
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
    Generates 6 charts for the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Overall performance by condition and model
    plt.figure(figsize=(12, 6))
    sns.barplot(x="condition", y="overall", hue="model", data=df, palette="magma")
    plt.title("R.E.C.I.P.E. Performance: Model and Condition Comparison", fontsize=15)
    plt.ylabel("Overall Score (Weighted)")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_results.png")
    plt.close()
    print(f"Saved: {output_dir}/overall_results.png")

    # 2. Metric breakdown heatmap
    metrics  = ["cause_accuracy", "solution_appropriateness", "scientific_accuracy"]
    pivot_df = df.groupby("condition")[metrics].mean()
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", cbar_kws={"label": "Score"})
    plt.title("Score Breakdown Across Metrics by Condition")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_heatmap.png")
    plt.close()
    print(f"Saved: {output_dir}/metric_heatmap.png")

    # 3. Scores by difficulty
    plt.figure(figsize=(10, 5))
    sns.barplot(x="difficulty", y="overall", hue="condition", data=df, palette="coolwarm")
    plt.title("Performance by Difficulty Level")
    plt.ylabel("Overall Score")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/by_difficulty.png")
    plt.close()
    print(f"Saved: {output_dir}/by_difficulty.png")

    # 4. Scores by recipe category
    plt.figure(figsize=(14, 6))
    sns.barplot(x="category", y="overall", hue="model", data=df, palette="viridis")
    plt.title("Performance by Recipe Category")
    plt.ylabel("Overall Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/by_category.png")
    plt.close()
    print(f"Saved: {output_dir}/by_category.png")

    # 5. Score distribution boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="condition", y="overall", hue="model", data=df, palette="Set2")
    plt.title("Score Distribution by Condition and Model")
    plt.ylabel("Overall Score")
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distribution.png")
    plt.close()
    print(f"Saved: {output_dir}/score_distribution.png")

    # 6. Hallucination rate by condition
    hall_df = hallucination_rate(df)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="condition", y="hallucinated", data=hall_df, palette="Reds")
    plt.title("Hallucination Rate by Condition")
    plt.ylabel("Rate (scientific accuracy < 0.2)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hallucination_rates.png")
    plt.close()
    print(f"Saved: {output_dir}/hallucination_rates.png")

    print(f"\nAll plots saved to {output_dir}/")


# ─────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────

def run_analysis():
    """
    Main function — loads results, prints report,
    saves CSVs and generates all 6 plots.
    """
    print("Loading results...")
    all_results = load_all_results(latest_only=True)

    if not all_results:
        print("No results to analyze yet.")
        print("Run experiments first: python main.py --mode full")
        return

    df = results_to_dataframe(all_results)
    print_report(df)
    save_analysis(df)

    print("\nGenerating visualizations...")
    generate_plots(df)


if __name__ == "__main__":
    run_analysis()