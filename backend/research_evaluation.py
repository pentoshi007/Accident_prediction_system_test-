#!/usr/bin/env python3
"""
=============================================================================
Research Evaluation Script — IEEE Paper Prerequisites
=============================================================================
AI-Based Accident Hotspot Prediction System Using DBSCAN & Random Forest

This script runs the full ML pipeline, generates all metrics, plots, and
tables required for an IEEE-format research paper.

Output directory:  backend/output/
  ├── plots/          High-resolution figures (.png, 300 DPI)
  ├── tables/         CSV / LaTeX tables
  └── metrics/        JSON metric dumps

Usage:
    cd backend
    python research_evaluation.py
=============================================================================
"""

import os
import sys
import json
import time
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_DIR, EDA_OUTPUT_DIR,
    SEVERITY_MAP, DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_TEST_SIZE,
    ARI_TIERS,
)

OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR      = os.path.join(OUTPUT_DIR, "tables")
METRICS_DIR     = os.path.join(OUTPUT_DIR, "metrics")

# Ensure output dirs exist
for d in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# IEEE-style matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEVERITY_LABELS = {1: "Slight Injury", 2: "Serious Injury", 3: "Fatal Injury"}
SEVERITY_COLORS = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
CLASS_NAMES = ["Slight", "Serious", "Fatal"]

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATA LOADING AND PREPROCESSING                         ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_preprocessing():
    """Run the preprocessing pipeline and return the processed DataFrame."""
    print("\n" + "=" * 70)
    print("  STEP 1 / 7 — Data Preprocessing")
    print("=" * 70)

    from scripts.preprocess import run as preprocess_run
    df = preprocess_run()
    return df


def load_processed_data():
    """Load the processed CSV into a DataFrame."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"  Loaded processed data: {len(df):,} rows, {len(df.columns)} columns")
    return df

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — EXPLORATORY DATA ANALYSIS PLOTS                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def generate_eda_plots(df):
    """Generate all EDA visualizations for the Results section."""
    print("\n" + "=" * 70)
    print("  STEP 2 / 7 — Exploratory Data Analysis Plots")
    print("=" * 70)

    # --- 2.1  Dataset Overview Table ---
    overview = {
        "Total Records": len(df),
        "Features Used": len(df.columns),
        "Severity Classes": 3,
        "Unique Areas": df["Area_accident_occured"].nunique() if "Area_accident_occured" in df.columns else "N/A",
        "Latitude Range": f"{df['Latitude'].min():.4f} – {df['Latitude'].max():.4f}",
        "Longitude Range": f"{df['Longitude'].min():.4f} – {df['Longitude'].max():.4f}",
    }
    overview_df = pd.DataFrame(list(overview.items()), columns=["Property", "Value"])
    overview_df.to_csv(os.path.join(TABLES_DIR, "dataset_overview.csv"), index=False)
    print("  Saved dataset_overview.csv")

    # --- 2.2  Severity Distribution (Bar + Pie) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sev_counts = df["Severity"].value_counts().sort_index()
    colors = [SEVERITY_COLORS[i] for i in sev_counts.index]
    labels = [SEVERITY_LABELS[i] for i in sev_counts.index]

    axes[0].bar(labels, sev_counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Severity Level")
    axes[0].set_ylabel("Number of Accidents")
    axes[0].set_title("(a) Severity Distribution")
    for i, (lbl, val) in enumerate(zip(labels, sev_counts.values)):
        axes[0].text(i, val + max(sev_counts.values)*0.02, str(val),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1].pie(sev_counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9},
                wedgeprops={"edgecolor": "black", "linewidth": 0.5})
    axes[1].set_title("(b) Severity Proportion")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_severity_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig1_severity_distribution.png")

    # --- 2.3  Hourly Accident Distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    hourly = df.groupby("Hour").size()
    hourly = hourly.reindex(range(24), fill_value=0)
    ax.fill_between(hourly.index, hourly.values, alpha=0.3, color="#3498db")
    ax.plot(hourly.index, hourly.values, "o-", color="#2c3e50", markersize=4, linewidth=1.5)
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Temporal Distribution of Accidents by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_hourly_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig2_hourly_distribution.png")

    # --- 2.4  Day-of-Week Distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = df.groupby("DayOfWeek").size().reindex(range(7), fill_value=0)
    bars = ax.bar(day_names, weekly.values, color="#8e44ad", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Accidents")
    ax.set_title("Accident Distribution by Day of Week")
    for bar, val in zip(bars, weekly.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(weekly.values)*0.01,
                str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_weekly_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig3_weekly_distribution.png")

    # --- 2.5  Weather Condition Distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    weather_counts = df["Weather_Binned"].value_counts()
    weather_colors = {"Clear": "#2ecc71", "Rain": "#3498db", "Fog": "#95a5a6",
                      "Snow": "#ecf0f1", "Wind": "#e67e22", "Other": "#9b59b6"}
    wc = [weather_colors.get(w, "#bdc3c7") for w in weather_counts.index]
    ax.barh(weather_counts.index, weather_counts.values, color=wc, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("Weather Condition")
    ax.set_title("Accident Distribution by Weather Condition")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_weather_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig4_weather_distribution.png")

    # --- 2.6  Severity vs Weather Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cross = pd.crosstab(df["Weather_Binned"], df["Severity"])
    cross.columns = [SEVERITY_LABELS.get(c, str(c)) for c in cross.columns]
    sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title("Severity Distribution Across Weather Conditions")
    ax.set_ylabel("Weather Condition")
    ax.set_xlabel("Severity Level")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_severity_weather_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig5_severity_weather_heatmap.png")

    # --- 2.7  Severity vs Light Conditions ---
    if "Light_conditions" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        cross_light = pd.crosstab(df["Light_conditions"], df["Severity"])
        cross_light.columns = [SEVERITY_LABELS.get(c, str(c)) for c in cross_light.columns]
        cross_light.plot(kind="bar", stacked=True, ax=ax,
                         color=["#2ecc71", "#f39c12", "#e74c3c"],
                         edgecolor="black", linewidth=0.3)
        ax.set_title("Severity vs Light Conditions")
        ax.set_xlabel("Light Condition")
        ax.set_ylabel("Count")
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig6_severity_light.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig6_severity_light.png")

    # --- 2.8  Top Accident Causes ---
    if "Cause_of_accident" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        cause_counts = df["Cause_of_accident"].value_counts().head(10)
        ax.barh(cause_counts.index[::-1], cause_counts.values[::-1],
                color="#16a085", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Number of Accidents")
        ax.set_title("Top 10 Causes of Accidents")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig7_top_causes.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig7_top_causes.png")

    # --- 2.9  Vehicle Type Distribution ---
    if "Type_of_vehicle" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        veh_counts = df["Type_of_vehicle"].value_counts().head(10)
        ax.barh(veh_counts.index[::-1], veh_counts.values[::-1],
                color="#2980b9", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Number of Accidents")
        ax.set_title("Top 10 Vehicle Types Involved in Accidents")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig8_vehicle_types.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig8_vehicle_types.png")

    # --- 2.10  Day/Night Accident Split ---
    fig, ax = plt.subplots(figsize=(5, 5))
    dn_counts = df["Is_Night"].value_counts()
    dn_labels = {0: "Daytime", 1: "Nighttime"}
    ax.pie([dn_counts.get(0, 0), dn_counts.get(1, 0)],
           labels=[dn_labels[0], dn_labels[1]],
           colors=["#f1c40f", "#2c3e50"],
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 10},
           wedgeprops={"edgecolor": "black", "linewidth": 0.5})
    ax.set_title("Day vs Night Accident Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig9_day_night.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig9_day_night.png")

    # --- 2.11  Correlation Heatmap (numeric features) ---
    numeric_cols = ["Hour", "DayOfWeek", "Is_Night", "Num_Vehicles", "Severity"]
    enc_cols = [c for c in df.columns if c.endswith("_Enc")]
    corr_cols = numeric_cols + enc_cols
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) > 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, ax=ax, linewidths=0.3,
                    annot_kws={"size": 7}, vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Matrix")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig10_correlation_heatmap.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig10_correlation_heatmap.png")

    # --- 2.12  Number of vehicles distribution ---
    if "Num_Vehicles" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        veh_dist = df["Num_Vehicles"].value_counts().sort_index()
        ax.bar(veh_dist.index.astype(str), veh_dist.values,
               color="#e74c3c", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Number of Vehicles Involved")
        ax.set_ylabel("Number of Accidents")
        ax.set_title("Distribution of Vehicles Involved Per Accident")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig11_num_vehicles.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig11_num_vehicles.png")

    # Save severity distribution table
    sev_table = pd.DataFrame({
        "Severity": [SEVERITY_LABELS[i] for i in sorted(sev_counts.index)],
        "Count": [sev_counts[i] for i in sorted(sev_counts.index)],
        "Percentage": [f"{sev_counts[i]/len(df)*100:.2f}%" for i in sorted(sev_counts.index)],
    })
    sev_table.to_csv(os.path.join(TABLES_DIR, "severity_distribution.csv"), index=False)
    print("  Saved severity_distribution.csv")

    # Save weather × severity cross-tab
    weather_sev = pd.crosstab(df["Weather_Binned"], df["Severity"], margins=True)
    weather_sev.columns = [SEVERITY_LABELS.get(c, str(c)) for c in weather_sev.columns]
    weather_sev.to_csv(os.path.join(TABLES_DIR, "weather_severity_crosstab.csv"))
    print("  Saved weather_severity_crosstab.csv")

    print("  [EDA] All EDA plots and tables generated.")

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — DBSCAN CLUSTERING ANALYSIS                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_clustering(df):
    """Run DBSCAN clustering and generate clustering evaluation plots."""
    print("\n" + "=" * 70)
    print("  STEP 3 / 7 — DBSCAN Spatial Clustering")
    print("=" * 70)

    from scripts.clustering import run as clustering_run
    cluster_summary = clustering_run()

    # Reload with Cluster_ID
    df = pd.read_csv(PROCESSED_DATA_PATH)

    labels = df["Cluster_ID"].values
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    n_clustered = int((labels != -1).sum())

    # --- 3.1  Cluster Scatter Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    noise = df[df["Cluster_ID"] == -1]
    clustered = df[df["Cluster_ID"] != -1]

    ax.scatter(noise["Longitude"], noise["Latitude"],
               c="gray", alpha=0.15, s=3, label="Noise/Outliers")

    unique_clusters = sorted(clustered["Cluster_ID"].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(unique_clusters), 1))
    for i, cid in enumerate(unique_clusters):
        mask = clustered["Cluster_ID"] == cid
        ax.scatter(clustered.loc[mask, "Longitude"],
                   clustered.loc[mask, "Latitude"],
                   c=[cmap(i % 20)], s=8, alpha=0.6,
                   label=f"Cluster {cid}" if i < 15 else None)

    # Plot centroids
    if cluster_summary is not None and len(cluster_summary) > 0:
        ax.scatter(cluster_summary["Centroid_Lon"],
                   cluster_summary["Centroid_Lat"],
                   marker="X", c="red", s=80, edgecolors="black",
                   linewidth=0.8, zorder=5, label="Centroids")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"DBSCAN Spatial Clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})")
    ax.legend(loc="upper left", fontsize=7, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig12_dbscan_clusters.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig12_dbscan_clusters.png")

    # --- 3.2  Cluster Size Distribution ---
    if cluster_summary is not None and len(cluster_summary) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_ids = cluster_summary["Cluster_ID"].values.astype(str)
        counts = cluster_summary["Incident_Count"].values
        sort_idx = np.argsort(counts)[::-1]
        ax.bar(range(len(counts)), counts[sort_idx],
               color="#3498db", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Cluster (sorted by size)")
        ax.set_ylabel("Number of Incidents")
        ax.set_title("Incident Count per DBSCAN Cluster")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(cluster_ids[sort_idx], rotation=45, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "fig13_cluster_sizes.png"), bbox_inches="tight")
        plt.close()
        print("  Saved fig13_cluster_sizes.png")

    # --- 3.3  Cluster Summary Table ---
    clustering_metrics = {
        "algorithm": "DBSCAN",
        "eps_radians": DBSCAN_EPS,
        "eps_km": round(DBSCAN_EPS * 6371.0, 2),
        "min_samples": DBSCAN_MIN_SAMPLES,
        "metric": "haversine",
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "n_clustered_points": n_clustered,
        "noise_ratio": round(n_noise / len(df), 4),
        "clustered_ratio": round(n_clustered / len(df), 4),
    }
    with open(os.path.join(METRICS_DIR, "clustering_metrics.json"), "w") as f:
        json.dump(clustering_metrics, f, indent=2)
    print("  Saved clustering_metrics.json")

    if cluster_summary is not None and len(cluster_summary) > 0:
        cluster_summary.to_csv(os.path.join(TABLES_DIR, "cluster_summary.csv"), index=False)
        print("  Saved cluster_summary.csv")

    return df, cluster_summary

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — RANDOM FOREST CLASSIFICATION & EVALUATION               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_classification(df):
    """Train RF, compute comprehensive metrics, and generate all model eval plots."""
    print("\n" + "=" * 70)
    print("  STEP 4 / 7 — Random Forest Classification & Evaluation")
    print("=" * 70)

    # Feature columns
    FEATURE_COLS = [
        "Hour", "DayOfWeek", "Is_Night", "Weather_Binned_Enc", "Num_Vehicles",
        "Type_of_vehicle_Enc", "Road_surface_type_Enc",
        "Road_surface_conditions_Enc", "Light_conditions_Enc",
        "Type_of_collision_Enc", "Cause_of_accident_Enc",
        "Road_allignment_Enc", "Types_of_Junction_Enc",
        "Lanes_or_Medians_Enc", "Driving_experience_Enc",
        "Age_band_of_driver_Enc", "Cluster_ID",
    ]
    TARGET_COL = "Severity"

    # Filter to clustered only
    df_model = df[df["Cluster_ID"] != -1].copy()
    available = [c for c in FEATURE_COLS if c in df_model.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        print(f"  Note: missing features (skipped): {missing}")

    X = df_model[available].values
    y = df_model[TARGET_COL].values
    classes = sorted(np.unique(y))

    print(f"  Features: {len(available)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Classes: {classes} → {[SEVERITY_LABELS.get(c, str(c)) for c in classes]}")
    print(f"  Distribution: { {int(u): int(c) for u, c in zip(*np.unique(y, return_counts=True))} }")

    # Stratified split
    stratify = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RF_TEST_SIZE, random_state=42, stratify=stratify
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train model
    print(f"  Training RF (n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}) ...")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    y_train_pred = clf.predict(X_train)

    # --------------- Compute Metrics ---------------
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    logloss = log_loss(y_test, y_proba, labels=classes)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  Test Accuracy      : {acc:.4f}              ║")
    print(f"  ║  Train Accuracy     : {train_acc:.4f}              ║")
    print(f"  ║  Precision (macro)  : {prec_macro:.4f}              ║")
    print(f"  ║  Recall (macro)     : {rec_macro:.4f}              ║")
    print(f"  ║  F1-Score (macro)   : {f1_macro:.4f}              ║")
    print(f"  ║  Cohen's Kappa      : {kappa:.4f}              ║")
    print(f"  ║  MCC                : {mcc:.4f}              ║")
    print(f"  ║  Log Loss           : {logloss:.4f}              ║")
    print(f"  ╚══════════════════════════════════════════════╝")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # --------------- Cross-Validation ---------------
    print("  Running 5-Fold Stratified Cross-Validation ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_precision = cross_val_score(clf, X, y, cv=skf, scoring="precision_macro", n_jobs=-1)
    cv_recall = cross_val_score(clf, X, y, cv=skf, scoring="recall_macro", n_jobs=-1)

    print(f"  CV Accuracy   : {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  CV F1 (macro) : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  CV Precision  : {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"  CV Recall     : {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

    # =========== SAVE ALL METRICS ===========
    all_metrics = {
        "model": "Random Forest Classifier",
        "hyperparameters": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "data_split": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": RF_TEST_SIZE,
        },
        "test_metrics": {
            "accuracy": round(acc, 4),
            "train_accuracy": round(train_acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "cohen_kappa": round(kappa, 4),
            "matthews_corr_coef": round(mcc, 4),
            "log_loss": round(logloss, 4),
        },
        "cross_validation_5fold": {
            "accuracy_mean": round(cv_accuracy.mean(), 4),
            "accuracy_std": round(cv_accuracy.std(), 4),
            "accuracy_per_fold": [round(v, 4) for v in cv_accuracy],
            "f1_macro_mean": round(cv_f1.mean(), 4),
            "f1_macro_std": round(cv_f1.std(), 4),
            "f1_per_fold": [round(v, 4) for v in cv_f1],
            "precision_macro_mean": round(cv_precision.mean(), 4),
            "precision_macro_std": round(cv_precision.std(), 4),
            "recall_macro_mean": round(cv_recall.mean(), 4),
            "recall_macro_std": round(cv_recall.std(), 4),
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                 for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
            for k, v in report.items()
        },
        "feature_columns": available,
        "severity_classes": {str(c): SEVERITY_LABELS.get(c, str(c)) for c in classes},
    }

    # Feature importances
    importances = dict(zip(available, clf.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    all_metrics["feature_importances"] = {k: round(v, 4) for k, v in sorted_imp}

    with open(os.path.join(METRICS_DIR, "classification_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("  Saved classification_metrics.json")

    # Save model artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "rf_model.joblib"))
    joblib.dump(importances, os.path.join(MODELS_DIR, "feature_importances.joblib"))

    # =========== GENERATE PLOTS ===========
    _plot_confusion_matrix(cm, classes)
    _plot_normalized_confusion_matrix(cm, classes)
    _plot_feature_importances(sorted_imp)
    _plot_roc_curves(y_test, y_proba, classes)
    _plot_precision_recall_curves(y_test, y_proba, classes)
    _plot_cv_boxplot(cv_accuracy, cv_f1, cv_precision, cv_recall)
    _plot_learning_curve(clf, X, y)
    _plot_class_wise_metrics(report, classes)
    _plot_prediction_confidence(y_proba, y_test, classes)

    # =========== GENERATE TABLES ===========
    _save_classification_tables(report, cm, classes, sorted_imp,
                                 cv_accuracy, cv_f1, cv_precision, cv_recall,
                                 all_metrics)

    return clf, importances, all_metrics


def _plot_confusion_matrix(cm, classes):
    """Plot raw confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES[:len(classes)],
                yticklabels=CLASS_NAMES[:len(classes)],
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig14_confusion_matrix.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig14_confusion_matrix.png")


def _plot_normalized_confusion_matrix(cm, classes):
    """Plot normalized (percentage) confusion matrix."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Oranges", ax=ax,
                xticklabels=CLASS_NAMES[:len(classes)],
                yticklabels=CLASS_NAMES[:len(classes)],
                linewidths=0.5, linecolor="gray",
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig15_confusion_matrix_normalized.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig15_confusion_matrix_normalized.png")


def _plot_feature_importances(sorted_imp):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [name for name, _ in sorted_imp]
    vals = [val for _, val in sorted_imp]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vals)))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1],
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    for i, (n, v) in enumerate(zip(names[::-1], vals[::-1])):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig16_feature_importances.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig16_feature_importances.png")


def _plot_roc_curves(y_test, y_proba, classes):
    """Plot multi-class ROC curves (One-vs-Rest)."""
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors_roc = ["#2ecc71", "#f39c12", "#e74c3c"]

    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)], linewidth=1.5,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    mean_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", linewidth=2,
            label=f"Macro avg (AUC = {mean_auc:.4f})")
    ax.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig17_roc_curves.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig17_roc_curves.png")

    # Save ROC AUC values
    roc_metrics = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_metrics[CLASS_NAMES[i]] = round(auc(fpr, tpr), 4)
    roc_metrics["macro_avg"] = round(mean_auc, 4)
    with open(os.path.join(METRICS_DIR, "roc_auc_scores.json"), "w") as f:
        json.dump(roc_metrics, f, indent=2)
    print("  Saved roc_auc_scores.json")


def _plot_precision_recall_curves(y_test, y_proba, classes):
    """Plot multi-class Precision-Recall curves."""
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors_pr = ["#2ecc71", "#f39c12", "#e74c3c"]

    for i in range(n_classes):
        precision_vals, recall_vals, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(recall_vals, precision_vals, color=colors_pr[i % len(colors_pr)],
                linewidth=1.5, label=f"{CLASS_NAMES[i]} (AP = {ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig18_precision_recall_curves.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig18_precision_recall_curves.png")


def _plot_cv_boxplot(cv_accuracy, cv_f1, cv_precision, cv_recall):
    """Box plot for cross-validation results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [cv_accuracy, cv_f1, cv_precision, cv_recall]
    labels_cv = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
    bp = ax.boxplot(data, patch_artist=True, labels=labels_cv)
    colors_box = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual fold scores
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.03, size=len(d))
        ax.plot(x, d, "ko", markersize=5, alpha=0.6)

    ax.set_ylabel("Score")
    ax.set_title("5-Fold Stratified Cross-Validation Results")
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig19_cv_boxplot.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig19_cv_boxplot.png")


def _plot_learning_curve(clf, X, y):
    """Plot learning curve (training vs validation accuracy vs training size)."""
    print("  Computing learning curve (may take a few seconds) ...")
    fig, ax = plt.subplots(figsize=(8, 5))

    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1, random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#3498db")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#e74c3c")
    ax.plot(train_sizes, train_mean, "o-", color="#3498db", linewidth=1.5,
            markersize=4, label="Training score")
    ax.plot(train_sizes, val_mean, "o-", color="#e74c3c", linewidth=1.5,
            markersize=4, label="Validation score")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — Random Forest")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig20_learning_curve.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig20_learning_curve.png")


def _plot_class_wise_metrics(report, classes):
    """Grouped bar chart of per-class precision, recall, F1."""
    fig, ax = plt.subplots(figsize=(8, 5))
    class_labels = CLASS_NAMES[:len(classes)]
    prec_vals = [report.get(c, {}).get("precision", 0) for c in class_labels]
    rec_vals = [report.get(c, {}).get("recall", 0) for c in class_labels]
    f1_vals = [report.get(c, {}).get("f1-score", 0) for c in class_labels]

    x = np.arange(len(class_labels))
    width = 0.25

    bars1 = ax.bar(x - width, prec_vals, width, label="Precision",
                   color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, rec_vals, width, label="Recall",
                   color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, f1_vals, width, label="F1-Score",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.set_ylim([0, 1.15])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig21_class_wise_metrics.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig21_class_wise_metrics.png")


def _plot_prediction_confidence(y_proba, y_test, classes):
    """Histogram of prediction confidence (max probability)."""
    max_proba = np.max(y_proba, axis=1)
    correct = (np.argmax(y_proba, axis=1) ==
               np.array([list(classes).index(y) for y in y_test]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(max_proba[correct], bins=30, alpha=0.7, color="#2ecc71",
            label="Correct predictions", edgecolor="black", linewidth=0.3)
    ax.hist(max_proba[~correct], bins=30, alpha=0.7, color="#e74c3c",
            label="Incorrect predictions", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Prediction Confidence (max probability)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Prediction Confidence")
    ax.legend()
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig22_prediction_confidence.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig22_prediction_confidence.png")


def _save_classification_tables(report, cm, classes, sorted_imp,
                                 cv_accuracy, cv_f1, cv_precision, cv_recall,
                                 all_metrics):
    """Save classification result tables as CSV."""

    # Per-class metrics table
    rows = []
    for i, cls in enumerate(CLASS_NAMES[:len(classes)]):
        r = report.get(cls, {})
        rows.append({
            "Class": cls,
            "Precision": round(r.get("precision", 0), 4),
            "Recall": round(r.get("recall", 0), 4),
            "F1-Score": round(r.get("f1-score", 0), 4),
            "Support": r.get("support", 0),
        })
    # Add macro/weighted avg
    for avg_type in ["macro avg", "weighted avg"]:
        r = report.get(avg_type, {})
        rows.append({
            "Class": avg_type.title(),
            "Precision": round(r.get("precision", 0), 4),
            "Recall": round(r.get("recall", 0), 4),
            "F1-Score": round(r.get("f1-score", 0), 4),
            "Support": r.get("support", 0),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(TABLES_DIR, "classification_report.csv"), index=False)
    print("  Saved classification_report.csv")

    # Confusion matrix table
    cm_df = pd.DataFrame(cm,
                         index=[f"True: {c}" for c in CLASS_NAMES[:len(classes)]],
                         columns=[f"Pred: {c}" for c in CLASS_NAMES[:len(classes)]])
    cm_df.to_csv(os.path.join(TABLES_DIR, "confusion_matrix.csv"))
    print("  Saved confusion_matrix.csv")

    # Feature importances table
    imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
    imp_df["Rank"] = range(1, len(imp_df) + 1)
    imp_df = imp_df[["Rank", "Feature", "Importance"]]
    imp_df.to_csv(os.path.join(TABLES_DIR, "feature_importances.csv"), index=False)
    print("  Saved feature_importances.csv")

    # CV results table
    cv_df = pd.DataFrame({
        "Fold": range(1, 6),
        "Accuracy": [round(v, 4) for v in cv_accuracy],
        "F1 (macro)": [round(v, 4) for v in cv_f1],
        "Precision (macro)": [round(v, 4) for v in cv_precision],
        "Recall (macro)": [round(v, 4) for v in cv_recall],
    })
    cv_df.loc[len(cv_df)] = ["Mean"] + [
        round(cv_accuracy.mean(), 4), round(cv_f1.mean(), 4),
        round(cv_precision.mean(), 4), round(cv_recall.mean(), 4)]
    cv_df.loc[len(cv_df)] = ["Std"] + [
        round(cv_accuracy.std(), 4), round(cv_f1.std(), 4),
        round(cv_precision.std(), 4), round(cv_recall.std(), 4)]
    cv_df.to_csv(os.path.join(TABLES_DIR, "cross_validation_results.csv"), index=False)
    print("  Saved cross_validation_results.csv")

    # Overall metrics summary table
    metrics_summary = pd.DataFrame({
        "Metric": [
            "Test Accuracy", "Train Accuracy",
            "Precision (macro)", "Precision (weighted)",
            "Recall (macro)", "Recall (weighted)",
            "F1-Score (macro)", "F1-Score (weighted)",
            "Cohen's Kappa", "Matthews Corr. Coeff.",
            "Log Loss",
            "CV Accuracy (mean ± std)",
            "CV F1 macro (mean ± std)",
        ],
        "Value": [
            all_metrics["test_metrics"]["accuracy"],
            all_metrics["test_metrics"]["train_accuracy"],
            all_metrics["test_metrics"]["precision_macro"],
            all_metrics["test_metrics"]["precision_weighted"],
            all_metrics["test_metrics"]["recall_macro"],
            all_metrics["test_metrics"]["recall_weighted"],
            all_metrics["test_metrics"]["f1_macro"],
            all_metrics["test_metrics"]["f1_weighted"],
            all_metrics["test_metrics"]["cohen_kappa"],
            all_metrics["test_metrics"]["matthews_corr_coef"],
            all_metrics["test_metrics"]["log_loss"],
            f"{all_metrics['cross_validation_5fold']['accuracy_mean']:.4f} ± {all_metrics['cross_validation_5fold']['accuracy_std']:.4f}",
            f"{all_metrics['cross_validation_5fold']['f1_macro_mean']:.4f} ± {all_metrics['cross_validation_5fold']['f1_macro_std']:.4f}",
        ],
    })
    metrics_summary.to_csv(os.path.join(TABLES_DIR, "metrics_summary.csv"), index=False)
    print("  Saved metrics_summary.csv")

    # Hyperparameters table
    hyper_df = pd.DataFrame({
        "Parameter": ["n_estimators", "max_depth", "class_weight",
                       "test_size", "random_state", "DBSCAN eps (rad)",
                       "DBSCAN min_samples", "DBSCAN metric"],
        "Value": [RF_N_ESTIMATORS, RF_MAX_DEPTH, "balanced",
                  RF_TEST_SIZE, 42, DBSCAN_EPS,
                  DBSCAN_MIN_SAMPLES, "haversine"],
    })
    hyper_df.to_csv(os.path.join(TABLES_DIR, "hyperparameters.csv"), index=False)
    print("  Saved hyperparameters.csv")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — ARI COMPUTATION & VISUALIZATION                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_ari(cluster_summary, importances):
    """Compute ARI scores and generate risk visualizations."""
    print("\n" + "=" * 70)
    print("  STEP 5 / 7 — Accident Risk Index (ARI)")
    print("=" * 70)

    from scripts.ari import (
        derive_weights, assign_tier, ENV_RISK_SCORES,
    )

    w1, w2, w3 = derive_weights(importances)
    print(f"  Weights: W1(severity)={w1}, W2(density)={w2}, W3(env)={w3}")

    clusters = cluster_summary.copy()

    # Severity Score (normalised 0-1)
    max_sev = clusters["Mean_Severity"].max()
    min_sev = clusters["Mean_Severity"].min()
    sev_range = max_sev - min_sev if max_sev != min_sev else 1.0
    clusters["Severity_Score"] = (clusters["Mean_Severity"] - min_sev) / sev_range

    # Density Score
    max_count = clusters["Incident_Count"].max()
    clusters["Density_Score"] = clusters["Incident_Count"] / max_count if max_count else 0

    # Environment Score
    clusters["Env_Score"] = clusters["Dominant_Weather"].map(ENV_RISK_SCORES).fillna(0.5)

    # ARI
    clusters["ARI_Score"] = (
        w1 * clusters["Severity_Score"] +
        w2 * clusters["Density_Score"] +
        w3 * clusters["Env_Score"]
    ).clip(0.0, 1.0)
    clusters["Risk_Tier"] = clusters["ARI_Score"].apply(assign_tier)

    # Save ARI results
    ari_metrics = {
        "weights": {"W1_severity": w1, "W2_density": w2, "W3_environment": w3},
        "ari_range": [round(float(clusters["ARI_Score"].min()), 4),
                      round(float(clusters["ARI_Score"].max()), 4)],
        "risk_tier_distribution": clusters["Risk_Tier"].value_counts().to_dict(),
        "env_risk_scores": ENV_RISK_SCORES,
        "ari_tier_thresholds": {k: list(v) for k, v in ARI_TIERS.items()},
    }
    with open(os.path.join(METRICS_DIR, "ari_metrics.json"), "w") as f:
        json.dump(ari_metrics, f, indent=2, default=str)
    print("  Saved ari_metrics.json")

    # ARI table
    ari_table = clusters[["Cluster_ID", "Centroid_Lat", "Centroid_Lon",
                           "Incident_Count", "Mean_Severity",
                           "Severity_Score", "Density_Score", "Env_Score",
                           "ARI_Score", "Risk_Tier", "Dominant_Weather"]].copy()
    for col in ["Centroid_Lat", "Centroid_Lon", "Mean_Severity", "Severity_Score",
                "Density_Score", "Env_Score", "ARI_Score"]:
        if col in ari_table.columns:
            ari_table[col] = ari_table[col].round(4)
    ari_table.to_csv(os.path.join(TABLES_DIR, "ari_results.csv"), index=False)
    print("  Saved ari_results.csv")

    # --- 5.1  Risk Tier Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    tier_counts = clusters["Risk_Tier"].value_counts()
    tier_order = ["Low", "Moderate", "Severe", "Critical"]
    tier_colors = {"Low": "#2ecc71", "Moderate": "#f1c40f",
                   "Severe": "#e67e22", "Critical": "#e74c3c"}

    present_tiers = [t for t in tier_order if t in tier_counts.index]
    bar_colors = [tier_colors[t] for t in present_tiers]
    bar_vals = [tier_counts[t] for t in present_tiers]

    axes[0].bar(present_tiers, bar_vals, color=bar_colors,
                edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Risk Tier")
    axes[0].set_ylabel("Number of Clusters")
    axes[0].set_title("(a) Risk Tier Distribution")
    for i, v in enumerate(bar_vals):
        axes[0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    # ARI Score histogram
    axes[1].hist(clusters["ARI_Score"], bins=15, color="#3498db",
                 edgecolor="black", linewidth=0.5, alpha=0.8)
    axes[1].set_xlabel("ARI Score")
    axes[1].set_ylabel("Number of Clusters")
    axes[1].set_title("(b) ARI Score Distribution")
    # Add tier boundary lines
    for tier, (lo, hi) in ARI_TIERS.items():
        if lo > 0:
            axes[1].axvline(x=lo, color=tier_colors[tier], linestyle="--",
                           linewidth=1.2, alpha=0.8)
            axes[1].text(lo + 0.01, axes[1].get_ylim()[1] * 0.9,
                        tier, fontsize=8, color=tier_colors[tier])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig23_ari_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig23_ari_distribution.png")

    # --- 5.2  ARI Component Breakdown ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_ids = clusters["Cluster_ID"].values
    x = np.arange(len(cluster_ids))
    width = 0.25

    ax.bar(x - width, clusters["Severity_Score"], width, label=f"Severity (W1={w1})",
           color="#e74c3c", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.bar(x, clusters["Density_Score"], width, label=f"Density (W2={w2})",
           color="#3498db", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.bar(x + width, clusters["Env_Score"], width, label=f"Environment (W3={w3})",
           color="#2ecc71", edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.plot(x, clusters["ARI_Score"], "kD-", markersize=5, linewidth=1.5,
            label="ARI Score", zorder=5)

    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Score (0–1)")
    ax.set_title("ARI Component Breakdown per Cluster")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_ids.astype(int), fontsize=8, rotation=45)
    ax.legend(loc="upper right")
    ax.set_ylim([0, 1.15])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig24_ari_components.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig24_ari_components.png")

    # --- 5.3  Risk Map (scatter colored by ARI) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        clusters["Centroid_Lon"], clusters["Centroid_Lat"],
        c=clusters["ARI_Score"], cmap="RdYlGn_r",
        s=clusters["Incident_Count"] * 2, edgecolors="black",
        linewidths=0.5, alpha=0.85, vmin=0, vmax=1
    )
    cbar = plt.colorbar(scatter, ax=ax, label="ARI Score")
    for _, row in clusters.iterrows():
        ax.annotate(f"C{int(row['Cluster_ID'])}", (row["Centroid_Lon"], row["Centroid_Lat"]),
                    fontsize=7, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Accident Black Spot Risk Map (size ∝ incident count)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig25_risk_map.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig25_risk_map.png")

    # Save ARI to models for API
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clusters, os.path.join(MODELS_DIR, "ari_data.joblib"))

    return clusters

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — COMPARATIVE ANALYSIS & ADDITIONAL RESULTS              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def comparative_analysis(df):
    """Compare RF against other classifiers for Table in paper."""
    print("\n" + "=" * 70)
    print("  STEP 6 / 7 — Comparative Model Analysis")
    print("=" * 70)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC

    FEATURE_COLS = [
        "Hour", "DayOfWeek", "Is_Night", "Weather_Binned_Enc", "Num_Vehicles",
        "Type_of_vehicle_Enc", "Road_surface_type_Enc",
        "Road_surface_conditions_Enc", "Light_conditions_Enc",
        "Type_of_collision_Enc", "Cause_of_accident_Enc",
        "Road_allignment_Enc", "Types_of_Junction_Enc",
        "Lanes_or_Medians_Enc", "Driving_experience_Enc",
        "Age_band_of_driver_Enc", "Cluster_ID",
    ]

    df_model = df[df["Cluster_ID"] != -1].copy()
    available = [c for c in FEATURE_COLS if c in df_model.columns]
    X = df_model[available].values
    y = df_model["Severity"].values
    stratify = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=RF_TEST_SIZE, random_state=42, stratify=stratify
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=RF_MAX_DEPTH, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }

    # Try to add SVM but only if dataset is small enough for reasonable runtime
    if len(X_train) < 15000:
        models["SVM (RBF)"] = SVC(kernel="rbf", class_weight="balanced",
                                   random_state=42, probability=True)

    results = []
    for name, model in models.items():
        print(f"  Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "Cohen's Kappa": round(kappa, 4),
            "Training Time (s)": round(train_time, 3),
        })
        print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  Time={train_time:.2f}s")

    comp_df = pd.DataFrame(results)
    comp_df = comp_df.sort_values("F1-Score", ascending=False)
    comp_df.to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)
    print("  Saved model_comparison.csv")

    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved model_comparison.json")

    # --- Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = comp_df["Model"].values
    x = np.arange(len(model_names))
    width = 0.18

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors_comp = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_comp)):
        vals = comp_df[metric].values
        offset = (i - len(metrics_to_plot)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=color, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison of Classification Models")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig26_model_comparison.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig26_model_comparison.png")

    return comp_df

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — RESEARCH SUMMARY OUTPUT                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def generate_research_summary(all_metrics, cluster_metrics_path, ari_metrics_path):
    """Generate a consolidated research summary JSON."""
    print("\n" + "=" * 70)
    print("  STEP 7 / 7 — Generating Research Summary")
    print("=" * 70)

    with open(cluster_metrics_path) as f:
        clust_m = json.load(f)
    with open(ari_metrics_path) as f:
        ari_m = json.load(f)

    summary = {
        "title": "AI-Based Accident Hotspot Prediction System Using DBSCAN and Random Forest",
        "dataset": {
            "name": "Road Accident Severity in India",
            "source": "Kaggle (s3programmer/road-accident-severity-in-india)",
            "total_records": all_metrics["data_split"]["train_size"] + all_metrics["data_split"]["test_size"],
            "target_variable": "Accident_severity (Slight/Serious/Fatal)",
            "features_used": len(all_metrics["feature_columns"]),
        },
        "methodology": {
            "preprocessing": "Geocoding, temporal feature engineering, label encoding (15 categoricals)",
            "clustering": {
                "algorithm": "DBSCAN with haversine distance",
                "parameters": f"eps={DBSCAN_EPS} rad (~{round(DBSCAN_EPS*6371, 1)} km), min_samples={DBSCAN_MIN_SAMPLES}",
                "clusters_found": clust_m["n_clusters"],
                "noise_ratio": clust_m["noise_ratio"],
            },
            "classification": {
                "algorithm": "Random Forest Classifier",
                "parameters": f"n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, class_weight=balanced",
                "features": len(all_metrics["feature_columns"]),
                "includes_cluster_id": "Yes (spatially-aware model)",
            },
            "risk_index": {
                "formula": "ARI = W1*Severity + W2*Density + W3*Environment",
                "weights": ari_m["weights"],
            },
        },
        "results": {
            "classification_performance": all_metrics["test_metrics"],
            "cross_validation": all_metrics["cross_validation_5fold"],
            "top_5_features": dict(list(all_metrics["feature_importances"].items())[:5]),
            "ari_risk_distribution": ari_m["risk_tier_distribution"],
        },
        "output_files": {
            "plots": sorted(os.listdir(PLOTS_DIR)),
            "tables": sorted(os.listdir(TABLES_DIR)),
            "metrics": sorted(os.listdir(METRICS_DIR)),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "research_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved research_summary.json")

    # Print final summary
    print("\n" + "=" * 70)
    print("  RESEARCH EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"\n  Generated files:")
    print(f"    Plots  ({len(os.listdir(PLOTS_DIR))} files): {PLOTS_DIR}")
    print(f"    Tables ({len(os.listdir(TABLES_DIR))} files): {TABLES_DIR}")
    print(f"    Metrics ({len(os.listdir(METRICS_DIR))} files): {METRICS_DIR}")
    print(f"\n  Key Results:")
    print(f"    Test Accuracy    : {all_metrics['test_metrics']['accuracy']}")
    print(f"    F1-Score (macro) : {all_metrics['test_metrics']['f1_macro']}")
    print(f"    Cohen's Kappa    : {all_metrics['test_metrics']['cohen_kappa']}")
    print(f"    MCC              : {all_metrics['test_metrics']['matthews_corr_coef']}")
    print(f"    CV Accuracy      : {all_metrics['cross_validation_5fold']['accuracy_mean']} ± {all_metrics['cross_validation_5fold']['accuracy_std']}")
    print(f"    Clusters Found   : {clust_m['n_clusters']}")
    print(f"    ARI Range        : {ari_m['ari_range']}")
    print()


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main():
    overall_start = time.time()

    print("\n" + "█" * 70)
    print("  AI-Based Accident Hotspot Prediction System")
    print("  IEEE Research Paper — Evaluation & Results Generation")
    print("█" * 70)

    # Step 1: Preprocessing
    run_preprocessing()

    # Step 2: EDA Plots
    df = load_processed_data()
    generate_eda_plots(df)

    # Step 3: DBSCAN Clustering
    df, cluster_summary = run_clustering(df)

    # Step 4: RF Classification (comprehensive)
    clf, importances, all_metrics = run_classification(df)

    # Step 5: ARI
    ari_clusters = run_ari(cluster_summary, importances)

    # Step 6: Comparative Analysis
    comparative_analysis(df)

    # Step 7: Research Summary
    generate_research_summary(
        all_metrics,
        os.path.join(METRICS_DIR, "clustering_metrics.json"),
        os.path.join(METRICS_DIR, "ari_metrics.json"),
    )

    elapsed = time.time() - overall_start
    print(f"  Total execution time: {elapsed:.1f}s")
    print("  Done.\n")


if __name__ == "__main__":
    main()
