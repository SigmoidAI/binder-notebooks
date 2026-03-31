"""
Helper utilities for the Automated Data Validation assignment.

This module provides data generation, visualization functions, and display
utilities to support learning about assertion-based and rule-based data
validation using pandas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_survey_dataset(
    n_samples: int = 300,
    error_rate: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a survey-like DataFrame with intentional data quality issues.

    Columns:
    - respondent_id : unique int IDs (with ~error_rate duplicate entries injected)
    - age           : int normally 18-80, with some negative values and values > 120
    - email         : strings like 'user@domain.com', some without '@' injected
    - score         : float 0-100, with some values outside that range injected
    - gender        : 'M'/'F'/'Other', with unexpected values like 'X' or '' injected
    - income        : positive float, with some negative values injected

    Args:
        n_samples:    Number of rows to generate.
        error_rate:   Fraction of rows that will carry quality issues (0–1).
        random_state: Random seed for reproducibility.

    Returns:
        pd.DataFrame with the six columns described above.
    """
    rng = np.random.default_rng(random_state)
    n_errors = max(1, int(n_samples * error_rate))

    # ── respondent_id ─────────────────────────────────────────────────────────
    ids = list(range(1, n_samples + 1))
    dup_positions = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    dup_sources   = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    for pos, src in zip(dup_positions, dup_sources):
        ids[pos] = ids[src]
    respondent_id = ids

    # ── age ───────────────────────────────────────────────────────────────────
    age = rng.integers(18, 81, size=n_samples).tolist()
    bad_age_pos = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    for i, pos in enumerate(bad_age_pos):
        # alternate between negative and > 120
        age[pos] = int(rng.integers(-20, 0)) if i % 2 == 0 else int(rng.integers(121, 200))

    # ── email ─────────────────────────────────────────────────────────────────
    domains = ["gmail.com", "yahoo.com", "outlook.com", "example.org"]
    email = [
        f"user{i}@{domains[i % len(domains)]}" for i in range(n_samples)
    ]
    bad_email_pos = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    bad_email_values = [
        "notanemail", "missingatsign.com", "nodomain@", "", "plainstring"
    ]
    for i, pos in enumerate(bad_email_pos):
        email[pos] = bad_email_values[i % len(bad_email_values)]

    # ── score ─────────────────────────────────────────────────────────────────
    score = rng.uniform(0.0, 100.0, size=n_samples).round(2).tolist()
    bad_score_pos = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    for i, pos in enumerate(bad_score_pos):
        score[pos] = float(rng.uniform(101, 150)) if i % 2 == 0 else float(rng.uniform(-50, -1))

    # ── gender ────────────────────────────────────────────────────────────────
    valid_genders = ["M", "F", "Other"]
    gender = rng.choice(valid_genders, size=n_samples).tolist()
    bad_gender_pos = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    bad_gender_values = ["X", "", "unknown", "N/A", "?"]
    for i, pos in enumerate(bad_gender_pos):
        gender[pos] = bad_gender_values[i % len(bad_gender_values)]

    # ── income ────────────────────────────────────────────────────────────────
    income = rng.uniform(20_000.0, 150_000.0, size=n_samples).round(2).tolist()
    bad_income_pos = rng.choice(n_samples, size=n_errors, replace=False).tolist()
    for pos in bad_income_pos:
        income[pos] = float(rng.uniform(-50_000, -1))

    return pd.DataFrame(
        {
            "respondent_id": respondent_id,
            "age":           age,
            "email":         email,
            "score":         score,
            "gender":        gender,
            "income":        income,
        }
    )


def visualize_validation_results(validation_report: dict) -> None:
    """
    Display a horizontal bar chart summarising a single validation report.

    Args:
        validation_report: dict of the form
            {rule_name: {'passed': bool, 'violations': int}}
            as produced by individual validation functions or
            create_validation_report.
    """
    if not validation_report:
        print("Empty validation report – nothing to plot.")
        return

    rules      = list(validation_report.keys())
    violations = []
    colors     = []

    for rule in rules:
        entry = validation_report[rule]
        v = entry.get("violations", 0) if entry.get("violations") is not None else 0
        violations.append(v)
        colors.append("#d9534f" if not entry.get("passed", True) else "#5cb85c")

    fig, ax = plt.subplots(figsize=(9, max(3, len(rules) * 0.6)))
    bars = ax.barh(rules, violations, color=colors, edgecolor="white", height=0.55)

    for bar, v in zip(bars, violations):
        ax.text(
            bar.get_width() + max(violations) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(v),
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Number of Violations", fontsize=11)
    ax.set_title("Validation Results – Violations per Rule", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#5cb85c", label="Passed"),
        Patch(facecolor="#d9534f", label="Failed"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.show()


def plot_validation_summary(results_list: list) -> None:
    """
    Stacked bar chart showing passed vs failed checks across multiple
    validation runs (e.g. before and after cleaning).

    Args:
        results_list: list of dicts, each with keys:
            'label'   – str, name for the run (e.g. 'Before Cleaning')
            'passed'  – int, number of checks that passed
            'failed'  – int, number of checks that failed
    """
    if not results_list:
        print("No results provided – nothing to plot.")
        return

    labels  = [r.get("label", f"Run {i+1}") for i, r in enumerate(results_list)]
    passed  = [r.get("passed", 0)            for r in results_list]
    failed  = [r.get("failed", 0)            for r in results_list]

    x      = np.arange(len(labels))
    width  = 0.5

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5))

    p1 = ax.bar(x, passed, width, label="Passed", color="#5cb85c", edgecolor="white")
    p2 = ax.bar(x, failed, width, bottom=passed, label="Failed",  color="#d9534f", edgecolor="white")

    ax.set_ylabel("Number of Checks", fontsize=11)
    ax.set_title("Validation Summary Across Runs", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar in p1:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h / 2,
                str(int(h)),
                ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            )
    for bar, base in zip(p2, passed):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + h / 2,
                str(int(h)),
                ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def display_violations(df: pd.DataFrame, mask: pd.Series, column: str, max_rows: int = 10) -> None:
    """
    Print the rows of *df* where *mask* is True (i.e. violation rows),
    highlighting the *column* of interest.

    Args:
        df:       The full DataFrame being validated.
        mask:     Boolean Series – True where a violation exists.
        column:   Name of the column being checked (for display purposes).
        max_rows: Maximum number of violation rows to show.
    """
    violations = df.loc[mask]
    n = len(violations)
    if n == 0:
        print(f"No violations found in column '{column}'.")
        return

    print(f"Found {n} violation(s) in column '{column}' (showing up to {max_rows}):")
    display_cols = [column] + [c for c in df.columns if c != column]
    print(violations[display_cols].head(max_rows).to_string(index=True))
