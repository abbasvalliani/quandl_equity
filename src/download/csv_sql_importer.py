import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


class CsvSqlImporter:
    def __init__(self, csv_file, table_name, host, database, username, password, date_columns):
        self.csv_file = csv_file
        self.table_name = table_name
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.date_columns = date_columns

    def do_import(self):
        # Read the CSV file
        df = pd.read_csv(self.csv_file, low_memory=False, parse_dates=self.date_columns)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print(f"Import CSV: {self.csv_file} with length: {len(df)} into table: {self.table_name}")

        engine = create_engine(f'mysql+pymysql://{self.username}:{self.password}@{self.host}/{self.database}')
        df.to_sql(name=self.table_name, con=engine, if_exists='replace', index=False)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(f"Provide exactly 6 arguments. Provided {len(sys.argv) - 1}")
        sys.exit(-1)

    csv_file = sys.argv[1]
    table_name = sys.argv[2]
    host = sys.argv[3]
    db = sys.argv[4]
    username = sys.argv[5]
    password = sys.argv[6]
    date_columns = sys.argv[7].split(',')
    importer = CsvSqlImporter(
        csv_file=csv_file,
        table_name=table_name,
        host=host,
        database=db,
        username=username,
        password=password,
        date_columns=date_columns
    )
    importer.do_import()
