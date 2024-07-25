import pandas as pd
import os
from sklearn.model_selection import train_test_split


class BaseModel:
    def __init__(self, data_file, model_path, train_mode=True):
        self.model_path = model_path
        self.data_file = data_file
        self.train_mode = train_mode
        self.model_data = None
        self.data = None
        self.model = None
        self.x, self.y, self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None, None, None

    @staticmethod
    def get_train_test_split():
        return 0.2

    def get_model_columns(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_feature_columns(self):
        columns = list(self.model_data.columns)
        for value in ['ticker', 'calendardate', 'result']:
            if value in columns:
                columns.remove(value)
        return columns

    def load_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def save_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_model(self):
        self.build_model()

    def load_data(self, num_rows=None):
        print(f"Reading datafile {self.data_file}")
        if num_rows == None:
            self.data = pd.read_csv(self.data_file)
        else:
            self.data = pd.read_csv(self.data_file, nrows=num_rows)

        print(f"Done reading file {self.data_file} and rows:{num_rows}")

        self.data['calendardate'] = pd.to_datetime(self.data['calendardate'])
        self.data['reportingquarter'] = self.data['calendardate'].dt.quarter

    def split_train_test(self, test_size=get_train_test_split()):
        # Split into training and testing sets
        print(f"Creating test and train datasets")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size,
                                                                                random_state=42)

    def prep_data(self):
        self.prep_data()

    def csv_save(self, data_frame, output_file):
        data_frame.to_csv(output_file, index=False)
