import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import load_and_split_data, clean_data, engineer_features

def preprocessing(raw_train, raw_val):
    clean_train, medians = clean_data(raw_train, is_train=True)
    clean_val, _ = clean_data(raw_val, is_train=False, equipment_weight_medians=medians)
    
    X_train, X_val = engineer_features(clean_train, clean_val)

    y_train = np.log1p(X_train['posted_rate'])
    y_val_true = X_val['posted_rate']

    drop_cols = ['posted_rate', 'load_id']
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    
    return X_train, y_train, X_val, y_val_true

def main():
    parser = argparse.ArgumentParser(description="Train freight rate prediction models.")
    parser.add_argument("--baseline", action="store_true", help="Run the Linear Regression baseline model")
    args = parser.parse_args()

    # 1. Load Data
    raw_train, raw_val = load_and_split_data('train-test.csv')
    
    # 2. Preprocess
    X_train, y_train, X_val, y_val_true = preprocessing(raw_train, raw_val)

    # 5. Model Selection
    if args.baseline:
        print("Training Linear Regression Baseline...")
        model = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
    else:
        print("Training default XGBoost Model...")
        # TODO: Add XGBoost model implementation
       
        return

    # 6. Train Model
    model.fit(X_train, y_train)

    # 7. Predict and Evaluate
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)

    mae = mean_absolute_error(y_val_true, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val_true, val_preds))

    print("-" * 30)
    if args.baseline:
        print(f"Baseline Linear Regression MAE:  ${mae:.2f}")
        print(f"Baseline Linear Regression RMSE: ${rmse:.2f}")
    else:
        print(f"XGBoost MAE:  ${mae:.2f}")
        print(f"XGBoost RMSE: ${rmse:.2f}")
    print("-" * 30)


if __name__ == "__main__":
    main()
