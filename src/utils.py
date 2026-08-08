import pandas as pd
import numpy as np

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

    # 4. Remove rate_per_mile outliers (
   
    if is_train and 'posted_rate' in df.columns and 'distance' in df.columns:
        df['rate_per_mile'] = df['posted_rate'] / df['distance']
        df = df[(df['rate_per_mile'] >= 0.5) & (df['rate_per_mile'] <= 8.0)].copy()
        df = df.drop(columns=['rate_per_mile']) # drop to prevent target leakage
        
    return df, equipment_weight_medians


