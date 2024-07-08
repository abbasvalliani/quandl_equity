#!/bin/sh

# Display all env variable
source .env

echo 'creating directory:' $EQUITY_ANALYTICS_DIR
mkdir -p $EQUITY_ANALYTICS_DIR

echo 'Running analytics prep'
python3 src/ml/analytics_prep.py $EQUITY_DATA_DIR $EQUITY_ANALYTICS_DIR