#!/bin/sh

# Display all env variable
source .env

echo 'Running analytics'
python3 src/ml/analytics_forecast.py "./model_checkpoint.keras" $EQUITY_ANALYTICS_DIR/forecast_data.csv $EQUITY_ANALYTICS_DIR