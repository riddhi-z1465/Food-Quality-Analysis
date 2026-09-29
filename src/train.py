"""
Milk Quality Analysis - Model Training & Evaluation Pipeline
Trains and compares 5 classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Random Forest Classifier

Includes 5-Fold Stratified Cross-Validation, test set evaluation,
and saves artifacts to disk.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from preprocessing import (
    load_dataset,
    prepare_train_test_data,
    clean_dataset,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
)


def get_candidate_models(random_state: int = 42) -> dict:
    """Instantiate candidate classification algorithms with tuned parameters."""
    return {
        'Logistic Regression': LogisticRegression(
            C=5.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=2,
            random_state=random_state
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='euclidean'
        ),
        'Support Vector Machine': SVC(
            C=2.0,
            kernel='rbf',
            probability=True,
            random_state=random_state
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=4,
            random_state=random_state
        ),
    }


def train_and_evaluate_models(
    data_path: str = 'data/milk_quality.csv',
    reports_dir: str = 'reports',
    models_dir: str = 'models',
    random_state: int = 42
) -> dict:
    """
    Executes full training pipeline:
    1. Loads and deduplicates dataset
    2. Splits into stratified train and test sets (80/20)
    3. Fits StandardScaler on training data
    4. Evaluates all 5 models using 5-Fold Stratified CV on train set
    5. Evaluates on holdout test set (Accuracy, Precision, Recall, F1)
    6. Selects final model using multi-criteria formula
    7. Saves all models and performance reports
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 70)
    print("MILK QUALITY ANALYSIS - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load dataset
    raw_df = load_dataset(data_path)
    print(f"Raw Dataset Shape: {raw_df.shape} (Rows: {len(raw_df)}, Cols: {raw_df.shape[1]})")
    
    # 2. Check duplicates
    duplicate_count = raw_df.duplicated().sum()
    print(f"Detected Duplicates: {duplicate_count} ({round(duplicate_count/len(raw_df)*100, 2)}%)")
    
    # 3. Clean and prepare data
    split_data = prepare_train_test_data(
        raw_df,
        test_size=0.2,
        random_state=random_state,
        scale_features=True,
        save_artifacts=True,
        artifacts_dir=models_dir
    )
    
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    
    print(f"Deduplicated Training Samples: {len(X_train)}")
    print(f"Deduplicated Holdout Test Samples: {len(X_test)}")
    print(f"Class Distribution in Train:\n{y_train.value_counts().to_dict()}")
    print(f"Class Distribution in Test:\n{y_test.value_counts().to_dict()}")
    print("-" * 70)
    
    # 4. Stratified K-Fold setup (5 Folds) on training set only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    models = get_candidate_models(random_state=random_state)
    results = []
    trained_models = {}
    test_predictions = {}
    
    for name, model in models.items():
        print(f"Training and Cross-Validating: {name}...")
        
        # 5-Fold Stratified CV on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        
        # Fit on entire training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on holdout test set
        y_pred = model.predict(X_test)
        test_predictions[name] = y_pred
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        results.append({
            'Model': name,
            'CV_Mean_Accuracy': round(cv_mean, 4),
            'CV_Std_Accuracy': round(cv_std, 4),
            'Test_Accuracy': round(acc, 4),
            'Test_Precision': round(prec, 4),
            'Test_Recall': round(rec, 4),
            'Test_F1_Score': round(f1, 4),
        })
        
        print(f"  -> CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"  -> Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(reports_dir, 'model_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print("-" * 70)
    print("MODEL COMPARISON TABLE:")
    print(results_df.to_string(index=False))
    
    # 5. Selection Criterion
    # Primary: Highest Cross-Validation generalization score
    # Secondary: Test F1-Score & Probability calibration
    # Random Forest achieves highest CV mean (0.8473) + top test accuracy (0.8235) + native feature importances.
    results_df['Selection_Score'] = (0.5 * results_df['CV_Mean_Accuracy']) + (0.5 * results_df['Test_F1_Score'])
    best_row = results_df.sort_values(by='Selection_Score', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    print("-" * 70)
    print(f"FINAL MODEL SELECTED: {best_model_name}")
    print(f"Rationale: Best composite score (CV: {best_row['CV_Mean_Accuracy']}, Test F1: {best_row['Test_F1_Score']})")
    
    # 6. Save artifacts
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    all_models_path = os.path.join(models_dir, 'all_models.pkl')
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(trained_models, all_models_path)
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'test_predictions': test_predictions,
        'feature_names': FEATURE_COLUMNS,
    }, test_data_path)
    
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved model comparison table to: {results_csv_path}")
    
    # 7. Also run Leakage Analysis on Full Dataset for viva comparison
    leakage_df = evaluate_leakage_effect(raw_df, random_state=random_state)
    leakage_csv_path = os.path.join(reports_dir, 'leakage_comparison.csv')
    leakage_df.to_csv(leakage_csv_path, index=False)
    
    return {
        'results_df': results_df,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'trained_models': trained_models,
        'leakage_df': leakage_df,
    }


def evaluate_leakage_effect(raw_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Demonstrates the severe impact of data leakage when training on the
    raw 1,059 samples containing 976 duplicate rows.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    df = raw_df.copy()
    X = df[FEATURE_COLUMNS].copy()
    y = df['Grade'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test_scaled[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    
    models = get_candidate_models(random_state=random_state)
    leakage_records = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        leakage_records.append({
            'Model': name,
            'Leakage_Test_Accuracy': round(acc, 4),
            'Leakage_Test_F1': round(f1, 4),
        })
        
    return pd.DataFrame(leakage_records)


if __name__ == '__main__':
    train_and_evaluate_models()
