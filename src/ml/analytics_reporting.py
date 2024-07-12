import sys

import pandas as pd
import os
from tensorflow.keras.models import load_model
from analytics_common import AnalyticsCommon
from constants import Constants


class ReportingModel:
    def __init__(self, forecast_output, pre_model_output, output_dir):
        self.forecast_output = forecast_output
        self.pre_model_output = pre_model_output
        self.output_dir = output_dir

    def report(self):
        pre_model = pd.read_csv(self.pre_model_output, low_memory=False,
                                parse_dates=['calendardate'])
        pre_model = pre_model.sort_values(by=['ticker', 'calendardate'], ascending=[True, False])
        pre_model = pre_model.drop_duplicates(subset='ticker', keep='first')
        forecast = pd.read_csv(self.forecast_output, low_memory=False)

        # add the last price
        output = pd.merge(forecast,
                          pre_model,
                          how='left',
                          left_on='ticker',
                          right_on='ticker')
        output = output[output['probability'].notna()]
        output = output[(output['last_price'] - output['price'])/output['price'] < 0.1]
        output = output.sort_values(by='probability', ascending=False)
        output.to_csv(os.path.join(self.output_dir, 'report.csv'), index=False)


def main():
    print("Reporting ...")
    if len(sys.argv) != 4:
        print(f"Please provide 3 arguments. Only {len(sys.argv) - 1} were provided")
        sys.exit(-1)

    forecast_output = sys.argv[1]
    pre_model_output = sys.argv[2]
    output_dir = sys.argv[3]

    model = ReportingModel(forecast_output=forecast_output, pre_model_output=pre_model_output, output_dir=output_dir)
    model.report()


main()
