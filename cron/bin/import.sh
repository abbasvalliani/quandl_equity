# Display all env variable
echo "Running process..."
DATA_DIR=/data/trader

mkdir -p $DATA_DIR

echo $QUANDL_API;
echo $DATA_DIR
echo $MYSQL_USER;
echo $MYSQL_PASSWORD;
echo $MYSQL_HOST;
echo $MYSQL_DATABASE;

LOG_FILE=/var/log/trader_cron.log
LOG_LEVEL=INFO

echo 'Running python import'
python3 /var/scripts/import.py $QUANDL_API $DATA_DIR $MYSQL_USER $MYSQL_PASSWORD $MYSQL_HOST $MYSQL_DATABASE $LOG_FILE $LOG_LEVEL

