"""
Milk Quality Analysis - Evaluation and Diagnostics Module
Computes detailed classification reports, confusion matrices,
model comparison charts, feature importance, and error analysis.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance

# Matplotlib formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

LABELS = ['Low', 'Medium', 'High']


def generate_evaluation_artifacts(
    models_dir: str = 'models',
    reports_dir: str = 'reports',
    figures_dir: str = 'reports/figures'
):
    """
    Executes full evaluation suite:
    - Generates confusion matrices for all candidate models
    - Creates model comparison bar chart
    - Computes feature importance (native and permutation)
    - Performs error analysis on misclassified test samples
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load artifacts
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    all_models_path = os.path.join(models_dir, 'all_models.pkl')
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    results_csv_path = os.path.join(reports_dir, 'model_results.csv')
    
    if not (os.path.exists(test_data_path) and os.path.exists(all_models_path)):
        raise FileNotFoundError("Training artifacts missing. Run 'python src/train.py' first.")
        
    test_data = joblib.load(test_data_path)
    trained_models = joblib.load(all_models_path)
    best_model = joblib.load(best_model_path)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    feature_names = test_data['feature_names']
    
    # Determine best model name
    best_name = None
    for name, model in trained_models.items():
        if type(model) == type(best_model):
            best_name = name
            break
    if best_name is None:
        best_name = 'Selected Best Model'
        
    print(f"Generating evaluation figures for {len(trained_models)} models...")
    
    # -------------------------------------------------------------
    # 2. Confusion Matrices for Each Model
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    model_names = list(trained_models.keys())
    for idx, name in enumerate(model_names):
        model = trained_models[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred):.3f} | F1: {f1_score(y_test, y_pred, average='weighted'):.3f}", fontsize=11, fontweight='bold', pad=10)
        axes[idx].grid(False)
        
        # Also save individual matrix
        fig_single, ax_single = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax_single, cmap='Blues')
        ax_single.set_title(f"Confusion Matrix - {name}", fontsize=12, fontweight='bold', pad=12)
        ax_single.grid(False)
        clean_name = name.lower().replace(' ', '_').replace('-', '_')
        single_path = os.path.join(figures_dir, f"confusion_matrix_{clean_name}.png")
        fig_single.tight_layout()
        fig_single.savefig(single_path, dpi=300)
        plt.close(fig_single)
        
    # Hide unused 6th subplot
    axes[5].axis('off')
    fig.suptitle("Confusion Matrix Comparison Across Models (Holdout Test Set)", fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout()
    all_cm_path = os.path.join(figures_dir, "all_confusion_matrices.png")
    fig.savefig(all_cm_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrices to: {all_cm_path}")
    
    # -------------------------------------------------------------
    # 3. Model Performance Comparison Chart
    # -------------------------------------------------------------
    if os.path.exists(results_csv_path):
        results_df = pd.read_csv(results_csv_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['CV_Mean_Accuracy', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1_Score']
        metric_labels = ['CV Mean Acc', 'Test Acc', 'Weighted Prec', 'Weighted Recall', 'Weighted F1']
        
        bar_df = results_df.set_index('Model')[metrics]
        bar_df.columns = metric_labels
        
        palette = ['#2b5c8f', '#3caea3', '#f6d55c', '#ed553b', '#6c5b7b']
        bar_df.plot(kind='bar', ax=ax, width=0.8, color=palette, edgecolor='black', linewidth=0.6)
        
        ax.set_title("Machine Learning Models Performance Comparison", fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax.set_xlabel("Classification Algorithm", fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)
        plt.xticks(rotation=15, ha='right', fontsize=10)
        
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=2, fontsize=7.5, rotation=90)
            
        fig.tight_layout()
        comparison_path = os.path.join(figures_dir, "model_comparison.png")
        fig.savefig(comparison_path, dpi=300)
        plt.close(fig)
        print(f"Saved model comparison chart to: {comparison_path}")
        
    # -------------------------------------------------------------
    # 4. Feature Importance (Native Random Forest & Permutation Best Model)
    # -------------------------------------------------------------
    # 4a. Random Forest Gini Importance
    rf_model = trained_models.get('Random Forest')
    if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
        bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='#1d3557', linewidth=0.8)
        
        ax.set_title("Random Forest Gini Feature Importance (Predictive Relative Weight)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Mean Impurity Decrease (Gini Importance)", fontsize=10)
        ax.set_xlim(0, max(importances) * 1.15)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va='center', fontsize=9, fontweight='semibold')
            
        fig.tight_layout()
        fi_path = os.path.join(figures_dir, "feature_importance.png")
        fig.savefig(fi_path, dpi=300)
        plt.close(fig)
        print(f"Saved Random Forest feature importance to: {fi_path}")
        
    # 4b. Permutation Importance for Selected Best Model (Works for KNN, SVM, etc.)
    perm_res = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, scoring='accuracy')
    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': perm_res.importances_mean,
        'Std': perm_res.importances_std
    }).sort_values('Mean_Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_df['Feature'], perm_df['Mean_Importance'], xerr=perm_df['Std'], color='#457b9d', edgecolor='#1d3557', capsize=4)
    ax.set_title(f"Permutation Feature Importance - {best_name} (Holdout Test Set)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Decrease in Accuracy when Feature is Permuted (Shuffled)", fontsize=10)
    fig.tight_layout()
    perm_path = os.path.join(figures_dir, "permutation_importance.png")
    fig.savefig(perm_path, dpi=300)
    plt.close(fig)
    print(f"Saved permutation feature importance to: {perm_path}")
    
    # -------------------------------------------------------------
    # 5. In-Depth Error Analysis
    # -------------------------------------------------------------
    y_pred_best = best_model.predict(X_test)
    misclassified_mask = (y_test.values != y_pred_best)
    
    # Reconstruct unscaled features for human-readable error analysis
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    X_test_unscaled = X_test.copy()
    X_test_unscaled[['pH', 'Temperature', 'Colour']] = scaler.inverse_transform(X_test[['pH', 'Temperature', 'Colour']])
    
    errors_df = X_test_unscaled[misclassified_mask].copy()
    errors_df['Actual_Grade'] = y_test.values[misclassified_mask]
    errors_df['Predicted_Grade'] = y_pred_best[misclassified_mask]
    
    error_csv_path = os.path.join(reports_dir, 'error_analysis.csv')
    errors_df.to_csv(error_csv_path, index=False)
    print(f"Total Test Samples: {len(y_test)} | Misclassified Samples: {len(errors_df)}")
    print(f"Saved error analysis data to: {error_csv_path}")
    
    # Error analysis visualization
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(errors_df) > 0:
        error_pairs = errors_df.groupby(['Actual_Grade', 'Predicted_Grade']).size().reset_index(name='Count')
        error_pairs['Pair'] = "Actual: " + error_pairs['Actual_Grade'] + "\nPred: " + error_pairs['Predicted_Grade']
        sns.barplot(data=error_pairs, x='Pair', y='Count', ax=ax, palette='Reds_r', edgecolor='black')
        ax.set_title(f"Misclassification Error Breakdown ({best_name})", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Count of Test Errors", fontsize=10)
        ax.set_xlabel("Confusion Pair", fontsize=10)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "Zero Misclassifications on Holdout Test Set!", ha='center', va='center', fontsize=12)
        ax.axis('off')
        
    fig.tight_layout()
    error_fig_path = os.path.join(figures_dir, "error_analysis.png")
    fig.savefig(error_fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved error analysis visualization to: {error_fig_path}")
    
    print("=" * 70)
    print("EVALUATION ARTIFACT GENERATION COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    
    return {
        'all_cm_path': all_cm_path,
        'comparison_path': comparison_path,
        'fi_path': fi_path,
        'perm_path': perm_path,
        'error_fig_path': error_fig_path,
        'errors_df': errors_df,
    }


if __name__ == '__main__':
    generate_evaluation_artifacts()
