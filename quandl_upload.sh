# Display all env variable
echo "Running process..."

source .env

mkdir -p $EQUITY_DATA_DIR
mkdir -p $LOG_DIR

echo $EQUITY_DATA_DIR
echo $MYSQL_USER;
echo $MYSQL_HOST;
echo $MYSQL_DATABASE;

LOG_FILE=$LOG_DIR/quandl_download.log
LOG_LEVEL=INFO

echo 'Running quandl download and mysql import'
python3 src/download/main.py $QUANDL_API $EQUITY_DATA_DIR $LOG_FILE $LOG_LEVEL $MYSQL_USER $MYSQL_PASSWORD $MYSQL_HOST $MYSQL_DATABASE

