"""
Milk Quality Analysis - Standalone Prediction Module
Provides programmatic and command-line interfaces for predicting
milk quality category (Low, Medium, High) given physical & chemical inputs.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    FEATURE_RANGES,
    preprocess_single_input,
    normalize_columns,
)


class MilkQualityPredictor:
    """Production-ready inference wrapper for Milk Quality classification."""
    
    def __init__(self, model_path: str = 'models/best_model.pkl', scaler_path: str = 'models/scaler.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}. Train model first using 'python src/train.py'.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler artifact not found at: {scaler_path}. Train model first using 'python src/train.py'.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = FEATURE_COLUMNS
        
    def validate_inputs(self, sample: dict) -> list[str]:
        """Verify inputs conform to physical and empirical parameter bounds."""
        warnings_and_errors = []
        for feature, bounds in FEATURE_RANGES.items():
            if feature not in sample:
                warnings_and_errors.append(f"Missing required parameter: {feature}")
                continue
                
            val = sample[feature]
            try:
                val = float(val)
            except (ValueError, TypeError):
                warnings_and_errors.append(f"Parameter '{feature}' must be numeric; got {type(val).__name__}")
                continue
                
            if val < bounds['min'] or val > bounds['max']:
                warnings_and_errors.append(
                    f"Parameter '{feature}' value {val} is outside empirical range [{bounds['min']}, {bounds['max']}]."
                )
        return warnings_and_errors

    def predict(self, sample: dict) -> dict:
        """
        Executes prediction pipeline on a single milk observation.
        Returns:
            dict containing predicted grade, class probabilities, and validated input.
        """
        validation_issues = self.validate_inputs(sample)
        if any("Missing" in err or "numeric" in err for err in validation_issues):
            raise ValueError(f"Input validation failed: {validation_issues}")
            
        # Preprocess features
        X_processed = preprocess_single_input(sample, scaler=self.scaler)
        
        # Predict class
        prediction = self.model.predict(X_processed)[0]
        
        # Calculate probabilities if supported
        probabilities = {}
        if hasattr(self.model, 'predict_proba'):
            raw_probs = self.model.predict_proba(X_processed)[0]
            classes = self.model.classes_
            probabilities = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(classes, raw_probs)
            }
            # Ensure keys Low, Medium, High are present
            for label in ['Low', 'Medium', 'High']:
                if label not in probabilities:
                    probabilities[label] = 0.0
                    
        return {
            'predicted_grade': str(prediction),
            'probabilities': probabilities,
            'features_submitted': sample,
            'validation_notices': validation_issues,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict Milk Quality (Low, Medium, High)")
    parser.add_argument('--ph', type=float, default=6.6, help="pH level (3.0 to 9.5)")
    parser.add_argument('--temperature', '--temp', type=float, default=37.0, help="Temperature in °C (34 to 90)")
    parser.add_argument('--taste', type=int, choices=[0, 1], default=1, help="Taste satisfaction flag (0 or 1)")
    parser.add_argument('--odor', type=int, choices=[0, 1], default=0, help="Odor satisfaction flag (0 or 1)")
    parser.add_argument('--fat', type=int, choices=[0, 1], default=1, help="Fat content flag (0 or 1)")
    parser.add_argument('--turbidity', type=int, choices=[0, 1], default=0, help="Turbidity condition flag (0 or 1)")
    parser.add_argument('--colour', '--color', type=int, default=255, help="Colour reflectance index (240 to 255)")
    
    args = parser.parse_args()
    
    input_sample = {
        'pH': args.ph,
        'Temperature': args.temperature,
        'Taste': args.taste,
        'Odor': args.odor,
        'Fat': args.fat,
        'Turbidity': args.turbidity,
        'Colour': args.colour,
    }
    
    try:
        predictor = MilkQualityPredictor()
        result = predictor.predict(input_sample)
        
        print("\n" + "=" * 50)
        print("MILK QUALITY PREDICTION RESULT")
        print("=" * 50)
        print(f"Predicted Quality Grade: {result['predicted_grade'].upper()}")
        print("-" * 50)
        if result['probabilities']:
            print("Confidence Distribution:")
            for grade, prob in sorted(result['probabilities'].items()):
                bar = "█" * int(prob * 30)
                print(f"  {grade:7s} : {prob*100:5.1f}%  |{bar}")
        print("-" * 50)
        print("Input Parameters:")
        for k, v in result['features_submitted'].items():
            print(f"  {k:12s}: {v}")
        if result['validation_notices']:
            print("\nNotices:")
            for note in result['validation_notices']:
                print(f"  * {note}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
