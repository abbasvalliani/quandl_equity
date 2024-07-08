#!/bin/sh

# Display all env variable
source .env

echo $MYSQL_USER;
echo $MYSQL_HOST;
echo $MYSQL_DATABASE;

echo 'Running csv import'
python3 src/download/csv_sql_importer.py $EQUITY_ANALYTICS_DIR/pre_model_ready.csv "pre_model_ready" $MYSQL_HOST $MYSQL_DATABASE $MYSQL_USER $MYSQL_PASSWORD "calendardate,real-rate-model-date,future_price_date,last_price_date,liq_date"
python3 src/download/csv_sql_importer.py $EQUITY_ANALYTICS_DIR/model_ready.csv "model_ready" $MYSQL_HOST $MYSQL_DATABASE $MYSQL_USER $MYSQL_PASSWORD "calendardate,reportperiod"
