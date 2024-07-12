#!/bin/sh

# Display all env variable
source .env

echo 'Running reporting'
python3 src/ml/analytics_reporting.py $EQUITY_ANALYTICS_DIR/forecast_results.csv $EQUITY_ANALYTICS_DIR/pre_model_ready.csv $EQUITY_ANALYTICS_DIR