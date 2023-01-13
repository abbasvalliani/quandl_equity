```
DISCLAIMER
==========
I have never coded in Python in my entire life. So, this is my first UGLY attempt at writing something
quickly to get this done. Only a few hours of code, but atleast the goal was achieved. 

WHAT DOES THIS DO
=================
If you subscribe to QUANDL, you can get access to the entire stock market data for the past 5 years.
This is critical if you're developing ML based equity models. The docker
images downloads QUANDL file every day and inserts it into a MYSQL database.

HOW DOES IT WORK
=================
Create a .env file in the directory with the following:

QUANDL_API=******
MYSQL_USER=myuser
MYSQL_PASSWORD=mypassword
MYSQL_DATABASE=mydatabase
MYSQL_HOST=trader-mysql

> To build and run:
docker-compose up -d

> To check logs:
cron\logs.cmd

> To access cron terminal:
cron\terminal.cmd
```
