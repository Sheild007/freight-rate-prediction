import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from utils import load_and_split_data, clean_data, engineer_features

def preprocessing(raw_train, raw_val, target_type='total'):
    
    # 1. Clean Data
    clean_train, medians = clean_data(raw_train, is_train=True)
    clean_val, _ = clean_data(raw_val, is_train=False, equipment_weight_medians=medians)
    
    # 2. Engineer Features
    X_train, X_val = engineer_features(clean_train, clean_val)

    # 3. Extract Targets
    if target_type == 'rpm':
        # Train on log(rate_per_mile)
        y_train = np.log1p(X_train['posted_rate'] / X_train['distance'])
    else:
        # Train on log(posted_rate)
        y_train = np.log1p(X_train['posted_rate'])
        
    y_val_true = X_val['posted_rate']
    val_distances = X_val['distance'].copy() # Save distances to multiply back later

    drop_cols = ['posted_rate', 'load_id']
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    
    return X_train, y_train, X_val, y_val_true, val_distances

def main():
    parser = argparse.ArgumentParser(description="Train freight rate prediction models.")
    parser.add_argument("-b", action="store_true", help="Run the Linear Regression baseline model")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning for XGBoost")
    parser.add_argument("--rpm", action="store_true", help="Predict Rate Per Mile instead of Total Rate")
    parser.add_argument("--lane_split", action="store_true", help="Split data geographically by lane instead of by time")
    args = parser.parse_args()

    # 1. Load Data
    split_type = 'lane' if args.lane_split else 'time'
    raw_train, raw_val = load_and_split_data('train-test.csv', split_type=split_type)
    
    # 2. Preprocess
    target_type = 'rpm' if args.rpm else 'total'
    X_train, y_train, X_val, y_val_true, val_distances = preprocessing(raw_train, raw_val, target_type)

    # 3. Model Selection
    if args.b:
        print(f"Training Linear Regression Baseline (Target: {target_type})...")
        model = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
    else:
        if args.tune:
            print(f"Initializing Tuning for XGBoost (Target: {target_type})...")
            param_grid = {
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            model = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10, 
                                       scoring='neg_mean_absolute_error', cv=3, random_state=42, verbose=1)
        else:
            print(f"Training default XGBoost Model (Target: {target_type})...")
            # Using the optimized parameters as the new default!
            model = xgb.XGBRegressor(
                n_estimators=1500, 
                learning_rate=0.01, 
                max_depth=4, 
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42, 
                n_jobs=-1
            )

    # 4. Train Model
    model.fit(X_train, y_train)

    if not args.b and args.tune:
        print(f"\nBest parameters found: {model.best_params_}")

    # 5. Predict and Evaluate
    val_preds_log = model.predict(X_val)
    
    if args.rpm:
        # Convert log(rpm) back to raw rpm, then multiply by distance to get total predicted rate
        val_preds_rpm = np.expm1(val_preds_log)
        val_preds = val_preds_rpm * val_distances
    else:
        # Standard conversion back to raw dollars
        val_preds = np.expm1(val_preds_log)

    mae = mean_absolute_error(y_val_true, val_preds)

    print("-" * 30)
    if args.b:
        print(f"Baseline Linear Regression MAE:  ${mae:.2f}")
    else:
        print(f"XGBoost MAE:  ${mae:.2f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
