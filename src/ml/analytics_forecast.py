import sys

import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
from analytics_common import AnalyticsCommon
from constants import Constants


class ForecastModel:
    def __init__(self, model_path, output_path, sequence_length):
        print(f"Loading model {model_path}")
        self.model = load_model(model_path)
        self.output_path = output_path
        self.sequence_length = sequence_length

    def load_data(self, data_file):
        return AnalyticsCommon.read_file(data_file)

    def preprocess_data(self, data):
        return AnalyticsCommon.preprocess(data=data.copy(), sequence_length=self.sequence_length)

    def cal_accuracy(self, data, expected_y):
        t, x, y = self.preprocess_data(data)
        baseline_accuracy = self.model.evaluate(x, expected_y, verbose=0)[1]
        print(f"Baseline accuracy of the results {baseline_accuracy}")
        if baseline_accuracy != 1:
            print("The results don't pass validation")
            sys.exit()

    def run_predictions(self, data_file):
        data = self.load_data(data_file)
        print(f"Running predictions for {len(data)} records.")
        t, x, y = self.preprocess_data(data)
        probability = self.model.predict(x)
        print("converted to binary predictions")
        binary_probability = (probability >= 0.5).astype(int)

        output = pd.DataFrame(t)
        output.columns = ["ticker"]
        output = output.drop_duplicates()
        output['probability'] = probability
        output['binary_probability'] = binary_probability
        print(f"Output records {len(output)}")
        print(f"Tickers with 10% increase {len(output)}")

        # validate results
        self.cal_accuracy(data, binary_probability)

        output.to_csv(os.path.join(self.output_path, 'forecast_results.csv'), index=False)

        return data


def main():
    print("Forecasting ...")
    if len(sys.argv) > 4:
        print(f"Please provide 3 arguments. Only {len(sys.argv)} were provided")
        sys.exit(-1)

    model_path = sys.argv[1]
    data_file = sys.argv[2]
    output_path = sys.argv[3]

    model = ForecastModel(model_path=model_path, output_path=output_path,
                          sequence_length=Constants.get_sequence_length())
    model.run_predictions(data_file)


main()
