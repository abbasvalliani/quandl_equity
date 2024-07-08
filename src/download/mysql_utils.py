import json
import sys
import time
import os
import shutil
import subprocess
import mysql.connector
import re
import logging
from utils import get_filename
from mysql.connector import errorcode
from pathlib import Path
from sys import exit


def update_table_import(table, last_updated, username, password, host, db):
    try:
        logging.info("META: %s, LastUpdated:%s" % (table, last_updated))
        mysqldb = mysql.connector.connect(
            user=username,
            password=password,
            host=host,
            db=db,
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )
        cursor = mysqldb.cursor()
        sql = "insert into META value (%s,%s) on duplicate key update last_updated = %s"
        val = (table, last_updated, last_updated)
        cursor.execute(sql, val)
        mysqldb.commit()
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise Exception("Access denied to MYSQL")
        elif e.errno == errorcode.ER_BAD_DB_ERROR:
            raise Exception("Database does not exist")
        else:
            raise e


def get_tables_to_update(download_meta, username, password, host, db):
    need_to_update_tables = []
    try:
        mysqldb = mysql.connector.connect(
            user=username,
            password=password,
            host=host,
            db=db,
            charset='utf8mb4',
            collation='utf8mb4_general_ci'
        )

        cursor = mysqldb.cursor()
        sql = "SELECT table_name, last_updated from META"
        rowcount = cursor.execute(sql)

        # get last updated times from database for each table
        table_db = {}
        if rowcount != 0:
            for row in cursor.fetchall():
                logging.info('DB INFO. Table Name:%s Last Updated:%s' % (row[0], row[1]))
                table_db[row[0]] = row[1]
        else:
            logging.warn("******** NO META DATA FOUND IN DATABASE ******")

        for table in download_meta:
            table_meta = download_meta[table]
            if table not in table_db:
                logging.info("Table %s will be updated (new download or not loaded to DB)" % (table))
                need_to_update_tables.append(table)
            else:
                last_refreshed_time = table_meta['last_refreshed_time']
                db_last_refreshed_time = table_db[table]
                if last_refreshed_time != db_last_refreshed_time:
                    logging.info("Table %s will be updated DB:'%s' DOWNLOAD:'%s'" % (
                    table, db_last_refreshed_time, last_refreshed_time))
                    need_to_update_tables.append(table)
        return need_to_update_tables
    except mysql.connector.Error as e:
        if e.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            raise Exception("Access denied to MYSQL")
        elif e.errno == errorcode.ER_BAD_DB_ERROR:
            raise Exception("Database does not exist")
        else:
            raise e


def sqlCommand(command):
    proc = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd='.', shell=True)
    res = proc.communicate()
    if proc.returncode == 1:
        print("res =", res)
        print("stderr =", res[1])
        logging.error("SQL Command error: %s %s", res, res[1])
    return proc.returncode


def create_table_meta(username, password, host, db):
    sql = '"CREATE TABLE IF NOT EXISTS META (table_name varchar(100), last_updated varchar(255), PRIMARY KEY (table_name))"'
    command = 'mysql -u%s -p%s -h%s --database=%s -e %s' % (username, password, host, db, sql)
    logging.debug("Executing SQL:" + command)

    proc = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode == 1:
        print("res =", res)
        print("stderr =", res[1])
        logging.error("SQL Command error: %s %s", res, res[1])
    return proc.returncode


def create_database(db, username, password, host):
    sql = '"CREATE DATABASE IF NOT EXISTS %s"' % (db)
    command = 'mysql -u%s -p%s -h%s -e %s' % (username, password, host, sql)
    logging.debug("Executing SQL " + command)

    proc = subprocess.Popen(
        command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    res = proc.communicate()
    if proc.returncode == 1:
        print("res =", res)
        print("stderr =", res[1])
        logging.error("SQL Command error: %s %s", res, res[1])
    return proc.returncode


def execSql(file, username, password, host, db):
    command = 'mysql -u%s -p%s -h%s --database=%s < %s' % (username, password, host, db, file)
    return sqlCommand(command)


def importCsv(file, username, password, host, db, table):
    header_row = ''

    # read the column headers
    with open(file) as f:
        header_row = f.readline().strip('\n')
        logging.debug('Table: %s Header row %s' % (table, header_row))
        f.close

    columns = header_row.split(',')

    # hack.. reserved name madeness
    for i in range(0, len(columns)):
        if columns[i] == "table":
            columns[i] = "table_1"

    column_import = ",".join(map(lambda a: '\'' + a + '\'', columns))
    if os.name == 'nt':
        file = re.escape(file)
    sql = "util.importTable('%s', {schema: '%s', table: '%s', dialect: 'csv-unix', skipRows: 1, showProgress: true, columns:[%s]})" % (
        file, db, table, column_import)
    command = 'mysqlsh --mysql -u%s -p%s -h%s --database=%s -e "%s"' % (username, password, host, db, sql)
    logging.debug("Running import command %s", sql)
    logging.info("Importing file into table %s" % (table))
    # return sqlCommand(command)

    popen = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    for c in iter(lambda: popen.stdout.read(1), b""):
        sys.stdout.buffer.write(c)
        f.buffer.write(c)
        if re.search('error', c, re.IGNORECASE):
            logging.error("Import Error. Table:%s Error: %s", table, c)
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return 0


if __name__ == "__main__":
    api_key = sys.argv[1]
    table = sys.argv[2]
    dir = sys.argv[3]
    bulk_fetch(api_key, table, dir)
    print('Done')
