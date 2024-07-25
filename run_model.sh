#!/bin/sh

# Display all env variable
source .env

echo 'Running analytics'
python3 src/ml/analytics_forecast.py $EQUITY_ANALYTICS_DIR/forecast_data.csv "./randomforest.joblib" $EQUITY_ANALYTICS_DIR/forecast_results.csv
#python3 src/ml/analytics_forecast.py $EQUITY_ANALYTICS_DIR/forecast_data.csv "cnn.keras" $EQUITY_ANALYTICS_DIR/forecast_results.csv