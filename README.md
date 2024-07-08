```
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

LOG_DIR=./temp_data/logs
EQUITY_DATA_DIR=./temp_data/equity_data
EQUITY_ANALYTICS_DIR=./temp_data/analytics_data
QUANDL_API=<use your key>
MYSQL_USER=trader
MYSQL_PASSWORD=<create one>
MYSQL_DATABASE=trader
MYSQL_HOST=localhost
MARIADB_ROOT_PASSWORD=<create one>

> create a conda environment (make sure conda is installed):
./env_setup.sh

> download equities data and import to mysql
./equity_download.sh

> run analytics preparation
./prep.sh

> import model ready data into mysql
./import_model_data.sh

> start training
./train.sh
```
