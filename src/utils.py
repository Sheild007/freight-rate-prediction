import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def load_and_split_data(filepath='train-test.csv'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Time-based split to avoid data leakage
    train_df = df[df['date'].dt.month <= 8].copy()
    val_df = df[df['date'].dt.month > 8].copy()
    
    return train_df, val_df
def clean_data(df, is_train=True, equipment_weight_medians=None):
   
    df = df.copy() 
    # 1. Fix negative weights by taking absolute value
    if 'weight' in df.columns:
        df['weight'] = df['weight'].abs()
        
        # 2. Impute missing weights using equipment medians
        if is_train:
            # If training, calculate the medians and save them to apply to test later
            equipment_weight_medians = df.groupby('equipment')['weight'].median()
            df['weight'] = df['weight'].fillna(df['equipment'].map(equipment_weight_medians))
        else:
            # If testing/validation, use the medians from the training set
            if equipment_weight_medians is not None:
                df['weight'] = df['weight'].fillna(df['equipment'].map(equipment_weight_medians))
                
    # 3. Handle missing market_index and quote_signal (forward fill then backward fill)
    if 'market_index' in df.columns:
        df = df.sort_values('date')
        df['market_index'] = df['market_index'].ffill().bfill()
    if 'quote_signal' in df.columns:
        df = df.sort_values('date')
        df['quote_signal'] = df['quote_signal'].ffill().bfill()
        
    df = df.sort_index()

    # 4. Remove rate_per_mile outliers 
    if is_train and 'posted_rate' in df.columns and 'distance' in df.columns:
        df['rate_per_mile'] = df['posted_rate'] / df['distance']
        df = df[(df['rate_per_mile'] >= 0.5) & (df['rate_per_mile'] <= 8.0)].copy()
        df = df.drop(columns=['rate_per_mile']) # drop to prevent target leakage
        
    return df, equipment_weight_medians
def target_encode(train_df, test_df, col, target, n_splits=5):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df[f"{col}_encoded"] = 0.0
    
    # K-fold encoding for train data
    for train_idx, val_idx in kf.split(train_df):
        mean_map = train_df.iloc[train_idx].groupby(col)[target].mean()
        train_df.loc[train_df.index[val_idx], f"{col}_encoded"] = train_df.iloc[val_idx][col].map(mean_map).values
        
    # Global mean to fill unseen categories
    global_mean = train_df[target].mean()
    train_df[f"{col}_encoded"] = train_df[f"{col}_encoded"].fillna(global_mean)
    
    # Encode test data using full train mapping
    full_mean_map = train_df.groupby(col)[target].mean()
    test_df[f"{col}_encoded"] = test_df[col].map(full_mean_map).fillna(global_mean)
    
    return train_df, test_df
def _create_base_features(df):
        df = df.copy()
        
        # 1. Temporal Features
        df['date_obj'] = pd.to_datetime(df['date'])
        df['month'] = df['date_obj'].dt.month
        df['day_of_week'] = df['date_obj'].dt.dayofweek
        df['day_of_month'] = df['date_obj'].dt.day
        
        # 2. Lane Feature
        df['lane'] = df['pickup'] + " -> " + df['delivery']
        
        # 3. Geographic Features (Lat/Lon differences)
        df['lat_abs_diff'] = (df['delivery_lat'] - df['pickup_lat']).abs()
        df['lon_abs_diff'] = (df['delivery_lon'] - df['pickup_lon']).abs()
        
        # 4. Interaction & Distance-Specific Features
        df["distance_x_market"] = df["distance"] * df["market_index"]
        df["weight_per_mile"] = df["weight"] / df["distance"]

        df["ton_miles"] = (df["weight"] / 2000) * df["distance"] # Standard freight metric
        df["log_distance"] = np.log1p(df["distance"]) # Captures economies of scale for longer trips
        df["is_short_haul"] = (df["distance"] <= 250).astype(int)
        df["is_long_haul"] = (df["distance"] >= 800).astype(int)
        
        # 5. Equipment One-Hot Encoding
        df = pd.get_dummies(df, columns=["equipment"], drop_first=True)
        # Convert booleans to int for ML models
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
                
        return df
def engineer_features(train_df, test_df):
    
    # Apply base feature generation to both train and test
    train_feat = _create_base_features(train_df)
    test_feat = _create_base_features(test_df)
    
    # 6. Target Encoding for high-cardinality categorical columns
    for col in ['lane', 'pickup', 'delivery']:
        train_feat, test_feat = target_encode(train_feat, test_feat, col, 'posted_rate')
        
    # Drop raw text/datetime columns that the model can't parse
    drop_cols = ['date', 'date_obj', 'pickup', 'delivery', 'lane']
    train_feat = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
    test_feat = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])
    
    return train_feat, test_feat
