DISCLAIMER
==========
I have never coded in Python in my entire life. So, this is my first UGLY attempt at writing something
quickly to get this done. Only a few hours of code, but atleast the goal was achieved. 

WHAT DOES THIS DO
=================
If you subscribe to QUANDL, you can get access to the entire stock market data for the past 30/40 years.
This is critical if you're developing ML based equity models. 

HOW DOES IT WORK
=================
Create a .env file in the directory with the following:

```
LOG_DIR=./temp_data/logs
EQUITY_DATA_DIR=./temp_data/equity_data
EQUITY_ANALYTICS_DIR=./temp_data/analytics_data
QUANDL_API=subscribe to core US fundamnetal data <https://data.nasdaq.com/publishers/SHARADAR>
FRED_API= get key at <https://fred.stlouisfed.org/docs/api/api_key.html>
MYSQL_USER=trader
MYSQL_PASSWORD=trader
MYSQL_DATABASE=trader
MYSQL_HOST=localhost
MARIADB_ROOT_PASSWORD=<set a password>
```

create a conda environment (make sure conda is installed) and install requirements:
>./env_setup.sh

download equities data
>./quandl_download.sh

upload to mysql data (optional)
>./quandl_upload.sh

run analytics preparation
>./prep.sh

start training
>./train.sh

run forecast
>./run_model.sh

run report
>./run_report.sh
