"""
Milk Quality Analysis - Exploratory Data Analysis (EDA) Module
Generates all 10 required academic figures:
1. Grade distribution count plot
2. pH distribution (histogram + KDE)
3. Temperature distribution (histogram + KDE)
4. Colour distribution
5. Boxplot of pH by Grade
6. Boxplot of Temperature by Grade
7. Boxplot of Colour by Grade
8. Feature distribution plots (all features multi-panel)
9. Correlation heatmap
10. Feature-vs-quality visualizations (binary attributes across grades)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from preprocessing import load_dataset, normalize_columns, clean_dataset

# Style configurations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
GRADE_ORDER = ['Low', 'Medium', 'High']
GRADE_PALETTE = {'Low': '#e63946', 'Medium': '#e9c46a', 'High': '#2a9d8f'}


def generate_all_eda_figures(data_path: str = 'data/milk_quality.csv', output_dir: str = 'reports/figures'):
    """Generate and save all 10 EDA visualizations to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    df_raw = load_dataset(data_path)
    df_clean, stats = clean_dataset(df_raw, drop_duplicates=True)
    
    print(f"Generating EDA plots for dataset ({len(df_clean)} deduplicated rows)...")

    # 1. Grade distribution count plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df_clean, x='Grade', order=GRADE_ORDER, palette=GRADE_PALETTE, ax=ax, edgecolor='black', linewidth=0.8)
    ax.set_title("1. Milk Quality Grade Class Distribution (Deduplicated)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Quality Category (Grade)", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3, fontsize=11, fontweight='semibold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "grade_distribution.png"), dpi=300)
    plt.close(fig)

    # 2. pH distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_clean['pH'], kde=True, bins=15, color='#1d3557', edgecolor='black', ax=ax)
    ax.set_title("2. Milk pH Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("pH Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.axvline(6.6, color='#e63946', linestyle='--', linewidth=1.5, label='Normal Fresh Milk pH (~6.6)')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ph_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Temperature distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_clean['Temperature'], kde=True, bins=15, color='#e76f51', edgecolor='black', ax=ax)
    ax.set_title("3. Milk Temperature Distribution (°C)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.axvline(37.0, color='#2a9d8f', linestyle='--', linewidth=1.5, label='Standard Fresh Milk Temp (~37°C)')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "temperature_distribution.png"), dpi=300)
    plt.close(fig)

    # 4. Colour distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df_clean, x='Colour', color='#457b9d', edgecolor='black', ax=ax)
    ax.set_title("4. Milk Colour Index Distribution (240 - 255)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Colour Reflectance Scale", fontsize=11)
    ax.set_ylabel("Observation Count", fontsize=11)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "colour_distribution.png"), dpi=300)
    plt.close(fig)

    # 5. Boxplot of pH by Grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_clean, x='Grade', y='pH', order=GRADE_ORDER, palette=GRADE_PALETTE, ax=ax, width=0.5, boxprops=dict(edgecolor='black'))
    sns.stripplot(data=df_clean, x='Grade', y='pH', order=GRADE_ORDER, color='black', alpha=0.5, jitter=0.2, ax=ax)
    ax.set_title("5. Milk pH Variation Across Quality Grades", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Quality Category (Grade)", fontsize=11)
    ax.set_ylabel("pH Level", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_ph_grade.png"), dpi=300)
    plt.close(fig)

    # 6. Boxplot of Temperature by Grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_clean, x='Grade', y='Temperature', order=GRADE_ORDER, palette=GRADE_PALETTE, ax=ax, width=0.5, boxprops=dict(edgecolor='black'))
    sns.stripplot(data=df_clean, x='Grade', y='Temperature', order=GRADE_ORDER, color='black', alpha=0.5, jitter=0.2, ax=ax)
    ax.set_title("6. Milk Temperature Variation Across Quality Grades", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Quality Category (Grade)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_temperature_grade.png"), dpi=300)
    plt.close(fig)

    # 7. Boxplot of Colour by Grade
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_clean, x='Grade', y='Colour', order=GRADE_ORDER, palette=GRADE_PALETTE, ax=ax, width=0.5, boxprops=dict(edgecolor='black'))
    sns.stripplot(data=df_clean, x='Grade', y='Colour', order=GRADE_ORDER, color='black', alpha=0.5, jitter=0.2, ax=ax)
    ax.set_title("7. Milk Colour Variation Across Quality Grades", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Quality Category (Grade)", fontsize=11)
    ax.set_ylabel("Colour Reflectance Value", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_colour_grade.png"), dpi=300)
    plt.close(fig)

    # 8. Feature distribution plots (multi-panel)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    
    for i, feat in enumerate(features):
        ax = axes[i]
        if feat in ['Taste', 'Odor', 'Fat', 'Turbidity']:
            sns.countplot(data=df_clean, x=feat, ax=ax, palette=['#e76f51', '#2a9d8f'], edgecolor='black')
            ax.set_ylabel("Count", fontsize=9)
        else:
            sns.histplot(df_clean[feat], kde=True, ax=ax, color='#264653', edgecolor='black')
            ax.set_ylabel("Density / Count", fontsize=9)
        ax.set_title(f"Distribution: {feat}", fontsize=11, fontweight='bold')
        ax.set_xlabel(feat, fontsize=10)
        
    axes[7].axis('off')
    fig.suptitle("8. Comprehensive Multi-Feature Distribution Summary", fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_distribution_plots.png"), dpi=300)
    plt.close(fig)

    # 9. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    df_numeric = df_clean[features].copy()
    df_numeric['Grade_Ordinal'] = df_clean['Grade'].map({'Low': 0, 'Medium': 1, 'High': 2})
    corr_matrix = df_numeric.corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-0.6,
        vmax=0.6,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title("9. Feature Correlation Matrix (Pearson r)", fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close(fig)

    # 10. Feature-vs-quality visualizations (Binary features breakdown by Grade)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    binary_cols = ['Taste', 'Odor', 'Fat', 'Turbidity']
    
    for i, col in enumerate(binary_cols):
        ax = axes[i]
        proportions = df_clean.groupby('Grade')[col].mean().reindex(GRADE_ORDER) * 100
        bars = ax.bar(GRADE_ORDER, proportions, color=[GRADE_PALETTE[g] for g in GRADE_ORDER], edgecolor='black', linewidth=0.8)
        ax.set_title(f"Optimal {col} % by Grade", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Quality Category", fontsize=10)
        ax.set_ylabel("Percentage with Flag = 1 (%)", fontsize=10)
        ax.set_ylim(0, 105)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha='center', fontsize=9, fontweight='semibold')
            
    fig.suptitle("10. Binary Sensor Attributes Prevalence by Milk Quality Grade", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_vs_quality.png"), dpi=300)
    plt.close(fig)

    print("All 10 EDA figures successfully generated in reports/figures/.")


if __name__ == '__main__':
    generate_all_eda_figures()
