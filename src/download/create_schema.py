import mysql.connector
from mysql.connector import errorcode
import getpass
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read connection parameters from console input
host = os.getenv('MYSQL_HOST')
root_user = input("Enter MySQL admin username: ")
root_password = getpass.getpass("Enter MySQL admin password: ")

# New schema (database) and user parameters from environment variables
new_schema = os.getenv('MYSQL_DATABASE')
new_user = os.getenv('MYSQL_USER')
new_password = os.getenv('MYSQL_PASSWORD')

# Initialize cursor as None
cursor = None
cnx = None

if not all([new_schema, new_user, new_password]):
    print("Environment variables for new schema, user, or password are not set.")
    exit(1)


# Connect to MySQL server
try:
    cnx = mysql.connector.connect(
        user=root_user,
        password=root_password,
        host=host,
        charset='utf8mb4',
        collation='utf8mb4_general_ci'
    )
    cursor = cnx.cursor()

    print("Connected to database")

    # Create new schema
    try:
        cursor.execute(f"CREATE DATABASE {new_schema}")
        print(f"Database {new_schema} created successfully.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DB_CREATE_EXISTS:
            print(f"Database {new_schema} already exists.")
        else:
            print(f"Failed creating database: {err}")
            exit(1)

    # Create new user and grant privileges
    try:
        cursor.execute(f"CREATE USER '{new_user}'@'%' IDENTIFIED BY '{new_password}'")
        print(f"User {new_user} created successfully.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_CANNOT_USER:
            print(f"Failed creating user: {err}")
            exit(1)
        else:
            print(f"User {new_user} already exists.")

    # Grant privileges to the new user on the new schema
    try:
        cursor.execute(f"GRANT ALL PRIVILEGES ON {new_schema}.* TO '{new_user}'@'%'")
        print(f"Granted all privileges on {new_schema} to {new_user}.")
    except mysql.connector.Error as err:
        print(f"Failed granting privileges: {err}")
        exit(1)

    # Commit the changes
    cnx.commit()

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
finally:
    if cursor:
        cursor.close()
    if cnx:
        cnx.close()
