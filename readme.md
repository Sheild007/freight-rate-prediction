# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:


## TODO

- [x] look at the data, check shapes, dtypes, missing values, weird stuff 
- [x] plot target distribution, check if it needs log transform
- [x] correlation heatmap  see what drives posted_rate the most
- [x] plot rates over time (daily/weekly/monthly) to catch trends and seasonality
- [x] box plots by equipment type
- [x] clean up the data fix missing and other errors
- [x] build temporal features
- [x] target-encode the pickup-delivery lanes 
- [x] one-hot encode equipment
- [x] add interaction features 
- [x] set up time-based split 
- [x] train a quick linear regression baseline, note down MAE/RMSE
- [x] train XGBoost with engineered features
- [x] tune hyperparams 
- [x] check feature importance, drop anything useless
- [x] run final model on validation.csv, fill out validation_predictions.csv
- [x] fill predicted_rate in december-chart-inputs.csv
- [x] run score.py to validate and generate the december chart



```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
