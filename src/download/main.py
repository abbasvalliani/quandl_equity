import json
import sys
import time
import os
import subprocess
import zipfile
import logging
import logging.handlers
from utils import mysql_delimiter_replace
from download import bulk_fetch
from create_ddl import *
from mysql_utils import *
from pathlib import Path
from sys import exit

sh_tables = ["INDICATORS", "ACTIONS", "EVENTS", "TICKERS", "SP500", "SF3A", "SF3B", "METRICS", "DAILY", "SF1", "SF2", "SF3", "SEP"]
ignore_index_tables = ["ACTIONS"]

updated_tables = []

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

def download_files(api_key, data_dir):
    download_meta = {}
    for table in sh_tables:
        result = bulk_fetch(api_key, table, data_dir)
        download_meta[table] = result
    return download_meta

# created the data directory
def import_data(api_key, data_dir, username, password, host, db):
    db_upload = (db is not None and username is not None and password is not None and host is not None)

    #
    # download data files from server
    #
    download_meta = download_files(api_key, data_dir)

    if not db_upload:
        logging.info("No database configuration provided.")
        sys.exit(0)

    #
    # create database if needed
    #
    ddl_dir = os.path.join(data_dir, 'ddl')

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(ddl_dir).mkdir(parents=True, exist_ok=True)

    # create database and meta table
    logging.info("Creating database if needed")
    retCode = create_database(db, username, password, host)
    if retCode == 1:
        logging.error("Unable to create database")
        exit(1)

    logging.info("Creating meta table if needed")
    retCode = create_table_meta(username, password, host, db)
    if retCode == 1:
        logging.error("Unable to create meta table")
        exit(1)


    #
    # add more tables if they have a zero count or missing tables
    #
    updated_tables = get_tables_to_update(download_meta, username, password, host, db)
    logging.info("Following tables need to be updated %s" % (updated_tables))

    #
    # create the schema from the INDICATORS file
    #
    schema_dictionary = {}
    indicator_zip_path = os.path.join(data_dir, 'INDICATORS.zip')
    with zipfile.ZipFile(indicator_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        indicator_file = os.path.join(data_dir, zip_ref.namelist()[0])
        schema_dictionary = load_table_definition(indicator_file)
        for updated_table in updated_tables:
            table_schema = create_ddl_and_index(updated_table, schema_dictionary)
            ddl_schema_file = os.path.join(ddl_dir, "%s.sql" % (updated_table))
            f = open(ddl_schema_file, "w")
            f.write(table_schema['ddl'])
            f.close()

            ddl_index_file = os.path.join(ddl_dir, "%s_index.sql" % (updated_table))
            f = open(ddl_index_file, "w")
            f.write(table_schema['index'])
            f.close()

    #
    # import files into the database
    #
    # import the tables into mysql
    for updated_table in updated_tables:
        last_refreshed_time = download_meta[updated_table]['last_refreshed_time']
        updated_table_path = os.path.join(data_dir, '%s.zip' % updated_table)
        with zipfile.ZipFile(updated_table_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            csv_file = os.path.join(data_dir, zip_ref.namelist()[0])
            logging.debug('Table: %s Extracted file %s' % (updated_table, csv_file))

            # replace the null
            logging.info("Starting to replace NULL's in table: %s" % (updated_table))
            mysql_delimiter_replace(csv_file)
            logging.info("Done with null replacements for table: %s" % (updated_table))
            ddl_schema_file = os.path.join(ddl_dir, "%s.sql" % (updated_table))
            ddl_index_file = os.path.join(ddl_dir, "%s_index.sql" % (updated_table))

            retCode = execSql(ddl_schema_file, username, password, host, db)
            if retCode:
                logging.error("Return code: %s. Unable to create table schema %s from file %s" % (
                retCode, updated_table, ddl_schema_file))
                exit(1)

            retCode = importCsv(csv_file, username, password, host, db, updated_table)
            if retCode:
                logging.error("Return code: %s. Unable to load table %s from file %s" % (retCode, updated_table, csv_file))
                exit(1)

            skip_indexing = 0
            for ignore_index_table in ignore_index_tables:
                if ignore_index_table == updated_table:
                    skip_indexing = 1
            if skip_indexing == 0:
                logging.info("Inserting index into Table: %s", (updated_table))
                retCode = execSql(ddl_index_file, username, password, host, db)
                if retCode:
                    logging.error("Return code: %s. Unable to index table %s from file %s" % (retCode, updated_table, ddl_index_file))
                    exit(1)
            logging.info("Updating table import success into Table: %s", (updated_table))
            update_table_import(updated_table, last_refreshed_time,username, password, host, db)

def main():
    api_key = sys.argv[1]
    data_dir = sys.argv[2]
    logfile = sys.argv[3]
    loglevel = sys.argv[4]

    username = password = host = db = None

    if len(sys.argv) > 5 and len(sys.argv) < 10:
        username = sys.argv[5]
        password = sys.argv[6]
        host = sys.argv[7]
        db = sys.argv[8]

    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

    logging.info('Started Import')

    import_data(api_key, data_dir, username, password, host, db)
    logging.info('Finished Import')

if __name__ == "__main__":
    main()
