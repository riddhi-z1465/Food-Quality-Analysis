"""
Milk Quality Analysis - Data Preprocessing Module
Handles data loading, column normalization, duplicate analysis,
cleaning, train-test splitting, and feature scaling.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Canonical column definitions
FEATURE_COLUMNS = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
NUMERICAL_COLUMNS = ['pH', 'Temperature', 'Colour']
BINARY_COLUMNS = ['Taste', 'Odor', 'Fat', 'Turbidity']
TARGET_COLUMN = 'Grade'

# Valid range definitions based on empirical dataset analysis
FEATURE_RANGES = {
    'pH': {'min': 3.0, 'max': 9.5, 'default': 6.6, 'step': 0.1, 'unit': 'pH'},
    'Temperature': {'min': 34.0, 'max': 90.0, 'default': 37.0, 'step': 1.0, 'unit': '°C'},
    'Taste': {'min': 0, 'max': 1, 'default': 1, 'options': [0, 1], 'labels': {0: 'Bad / Suboptimal (0)', 1: 'Good / Optimal (1)'}},
    'Odor': {'min': 0, 'max': 1, 'default': 0, 'options': [0, 1], 'labels': {0: 'Bad / Suboptimal (0)', 1: 'Good / Optimal (1)'}},
    'Fat': {'min': 0, 'max': 1, 'default': 1, 'options': [0, 1], 'labels': {0: 'Low / Inadequate (0)', 1: 'High / Optimal (1)'}},
    'Turbidity': {'min': 0, 'max': 1, 'default': 0, 'options': [0, 1], 'labels': {0: 'Low / Normal (0)', 1: 'High / Turbid (1)'}},
    'Colour': {'min': 240, 'max': 255, 'default': 255, 'step': 1, 'unit': 'Scale (240-255)'},
}

GRADE_MAPPING = {'Low': 0, 'Medium': 1, 'High': 2}
GRADE_INVERSE_MAPPING = {0: 'Low', 1: 'Medium', 2: 'High'}


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file and normalize column names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    df = normalize_columns(df)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names:
    - Strip leading/trailing whitespaces (e.g. 'Fat ' -> 'Fat')
    - Correct common spelling variants (e.g. 'Temprature' -> 'Temperature')
    - Normalize target column casing (e.g. 'low' -> 'Low')
    """
    df = df.copy()
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Rename known spelling variations
    rename_dict = {}
    for col in df.columns:
        if col.lower() in ['temprature', 'temperature']:
            rename_dict[col] = 'Temperature'
        elif col.lower() == 'fat':
            rename_dict[col] = 'Fat'
        elif col.lower() == 'turbidity':
            rename_dict[col] = 'Turbidity'
        elif col.lower() in ['colour', 'color']:
            rename_dict[col] = 'Colour'
        elif col.lower() == 'grade':
            rename_dict[col] = 'Grade'
        elif col.lower() == 'taste':
            rename_dict[col] = 'Taste'
        elif col.lower() in ['odor', 'odour']:
            rename_dict[col] = 'Odor'
        elif col.lower() == 'ph':
            rename_dict[col] = 'pH'
            
    df = df.rename(columns=rename_dict)
    
    # Capitalize target Grade labels if present
    if 'Grade' in df.columns:
        df['Grade'] = df['Grade'].astype(str).str.capitalize()
        
    return df


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """
    Examine dataset dimensions, data types, missing values,
    duplicate rows, and value boundaries.
    """
    df_norm = normalize_columns(df)
    total_rows, total_cols = df_norm.shape
    missing_counts = df_norm.isnull().sum().to_dict()
    duplicate_count = int(df_norm.duplicated().sum())
    unique_rows = total_rows - duplicate_count
    
    feature_stats = {}
    for col in df_norm.columns:
        feature_stats[col] = {
            'dtype': str(df_norm[col].dtype),
            'unique_count': int(df_norm[col].nunique()),
            'min': float(df_norm[col].min()) if pd.api.types.is_numeric_dtype(df_norm[col]) else None,
            'max': float(df_norm[col].max()) if pd.api.types.is_numeric_dtype(df_norm[col]) else None,
        }
        
    return {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'duplicate_count': duplicate_count,
        'duplicate_percentage': round((duplicate_count / total_rows) * 100, 2),
        'unique_rows': unique_rows,
        'missing_counts': missing_counts,
        'feature_stats': feature_stats,
    }


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Execute comprehensive data cleaning:
    - Normalizes column names
    - Handles missing values if present (fills or drops)
    - Optionally removes exact duplicate rows
    - Validates feature ranges
    Returns cleaned DataFrame and cleaning audit statistics.
    """
    df_cleaned = normalize_columns(df)
    initial_rows = len(df_cleaned)
    
    # Missing values check
    missing_total = int(df_cleaned.isnull().sum().sum())
    if missing_total > 0:
        df_cleaned = df_cleaned.dropna().reset_index(drop=True)
        
    # Duplicate records check and handling
    duplicates_found = int(df_cleaned.duplicated().sum())
    if drop_duplicates:
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    
    final_rows = len(df_cleaned)
    
    stats = {
        'initial_rows': initial_rows,
        'duplicates_removed': duplicates_found if drop_duplicates else 0,
        'final_rows': final_rows,
        'missing_values_handled': missing_total,
        'class_distribution': df_cleaned['Grade'].value_counts().to_dict() if 'Grade' in df_cleaned.columns else {},
    }
    
    return df_cleaned, stats


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    save_artifacts: bool = True,
    artifacts_dir: str = 'models'
):
    """
    Prepares train and test splits with strict leakage prevention:
    - Splits raw features and target using StratifiedKFold logic
    - Fits StandardScaler ONLY on X_train numerical columns
    - Transforms both X_train and X_test
    - Encodes target Grade into integers (0: Low, 1: Medium, 2: High)
    - Optionally saves scaler and label encoder to disk
    """
    df_clean, _ = clean_dataset(df, drop_duplicates=True)
    
    X = df_clean[FEATURE_COLUMNS].copy()
    y = df_clean[TARGET_COLUMN].copy()
    
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Fit scaler only on training data numerical features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        # Scale only numerical columns to preserve interpretability of binary flags
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
        X_test_scaled[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
    if save_artifacts and scaler is not None:
        os.makedirs(artifacts_dir, exist_ok=True)
        scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
    }


def preprocess_single_input(raw_input: dict, scaler=None) -> pd.DataFrame:
    """
    Transforms a single sample dictionary into a preprocessed DataFrame
    ready for model inference.
    """
    # Create DataFrame from input
    df_input = pd.DataFrame([raw_input])
    df_input = normalize_columns(df_input)
    
    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            raise ValueError(f"Missing required feature: {col}")
            
    df_ordered = df_input[FEATURE_COLUMNS].copy()
    
    # Apply fitted scaler to numerical columns if provided
    if scaler is not None:
        df_ordered[NUMERICAL_COLUMNS] = scaler.transform(df_ordered[NUMERICAL_COLUMNS])
        
    return df_ordered
