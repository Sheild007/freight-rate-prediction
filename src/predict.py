import pandas as pd
import numpy as np
import xgboost as xgb
from utils import clean_data, engineer_features

def main():
    print("Loading datasets...")
    raw_train = pd.read_csv('train-test.csv')
    raw_val = pd.read_csv('validation.csv')
    raw_dec = pd.read_csv('december-chart-inputs.csv')
  
    print("Preprocessing Validation Data...")
    
    clean_train, medians = clean_data(raw_train, is_train=True)
    clean_val, _ = clean_data(raw_val, is_train=False, equipment_weight_medians=medians)
    
    X_train_val, X_val = engineer_features(clean_train, clean_val)
    
    print("Preprocessing December Data...")
    city_map = pd.concat([
        raw_train[['pickup', 'pickup_lat', 'pickup_lon']].rename(columns={'pickup': 'city', 'pickup_lat': 'lat', 'pickup_lon': 'lon'}),
        raw_train[['delivery', 'delivery_lat', 'delivery_lon']].rename(columns={'delivery': 'city', 'delivery_lat': 'lat', 'delivery_lon': 'lon'})
    ]).drop_duplicates('city').set_index('city')
    
    raw_dec['pickup_lat'] = raw_dec['pickup'].map(city_map['lat'])
    raw_dec['pickup_lon'] = raw_dec['pickup'].map(city_map['lon'])
    raw_dec['delivery_lat'] = raw_dec['delivery'].map(city_map['lat'])
    raw_dec['delivery_lon'] = raw_dec['delivery'].map(city_map['lon'])
    
    raw_dec['market_index'] = raw_val['market_index'].dropna().iloc[-1]
    raw_dec['quote_signal'] = raw_val['quote_signal'].dropna().iloc[-1]
    
    clean_dec, _ = clean_data(raw_dec, is_train=False, equipment_weight_medians=medians)
    _, X_dec = engineer_features(clean_train, clean_dec)
    
    
    print("Training final XGBoost model on 100% of training data...")
    # Target is Rate Per Mile
    y_train = np.log1p(X_train_val['posted_rate'] / X_train_val['distance'])
    
    drop_cols = ['posted_rate', 'load_id']
    X_train_final = X_train_val.drop(columns=[c for c in drop_cols if c in X_train_val.columns])
    
    model = xgb.XGBRegressor(
        n_estimators=1500, 
        learning_rate=0.01, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train_final, y_train)
    
   
    print("Predicting Validation Data...")
    X_val_final = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])
    val_distances = X_val['distance']
    
   
    for col in X_train_final.columns:
        if col not in X_val_final.columns:
            X_val_final[col] = 0
    X_val_final = X_val_final[X_train_final.columns]
    
    val_preds_rpm_log = model.predict(X_val_final)
    val_preds = np.expm1(val_preds_rpm_log) * val_distances
    
    val_template = pd.read_csv('validation-predictions-template.csv')
    val_template['predicted_rate'] = val_preds.round(2)
    val_template.to_csv('validation-predictions.csv', index=False)
    print(" Saved validation-predictions.csv")
    
   
    print("Predicting December Data...")
    X_dec_final = X_dec.drop(columns=[c for c in drop_cols if c in X_dec.columns])
    dec_distances = X_dec['distance']
    
    for col in X_train_final.columns:
        if col not in X_dec_final.columns:
            X_dec_final[col] = 0
    X_dec_final = X_dec_final[X_train_final.columns]
    
    dec_preds_rpm_log = model.predict(X_dec_final)
    dec_preds = np.expm1(dec_preds_rpm_log) * dec_distances
    
    fresh_dec = pd.read_csv('december-chart-inputs.csv')
    fresh_dec['predicted_rate'] = dec_preds.round(2)
    fresh_dec.to_csv('december-chart-inputs-completed.csv', index=False)
    print(" Saved december-chart-inputs-completed.csv")

if __name__ == "__main__":
    main()
