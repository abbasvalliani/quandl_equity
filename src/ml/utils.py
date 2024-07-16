import zipfile
import pandas as pd

class Utils:
    @staticmethod
    def unzip(zip_file_path, num_rows):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if not file_list:
                raise ValueError("The zip archive is empty.")

            first_file = file_list[0]

            with zip_ref.open(first_file) as file:
                print(f"Reading file {first_file}")
                if num_rows is None:
                    df = pd.read_csv(file)
                else:
                    df = pd.read_csv(file, nrows=num_rows)
                print(f"Done Reading file {first_file}")
                return df

    @staticmethod
    def load_pandas(file, num_rows):
        df = pd.read_csv(file, nrows=num_rows)
        return df
