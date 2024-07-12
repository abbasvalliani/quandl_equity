#!/bin/sh

# Display all env variable
source .env

echo $MYSQL_USER;
echo $MYSQL_HOST;
echo $MYSQL_DATABASE;

echo 'Running csv import'
python3 src/download/csv_sql_importer.py $EQUITY_ANALYTICS_DIR/report.csv "report" $MYSQL_HOST $MYSQL_DATABASE $MYSQL_USER $MYSQL_PASSWORD "calendardate,last_price_date"
