#!/bin/sh

# Display all env variable
source .env

echo 'Running analytics'
python3 src/ml/analytics_train.py $EQUITY_ANALYTICS_DIR/model_ready.csv "randomforest.joblib"
#python3 src/ml/analytics_train.py $EQUITY_ANALYTICS_DIR/model_ready.csv "cnn.keras"