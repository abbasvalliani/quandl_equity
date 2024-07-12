# Display all env variable
echo "Running process..."

source .env

mkdir -p $EQUITY_DATA_DIR
mkdir -p $LOG_DIR

echo $EQUITY_DATA_DIR

LOG_FILE=$LOG_DIR/quandl_download.log
LOG_LEVEL=INFO

echo 'Running quandl download'
python3 src/download/main.py $QUANDL_API $EQUITY_DATA_DIR $LOG_FILE $LOG_LEVEL

