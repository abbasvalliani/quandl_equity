import sys

import pandas as pd
import os
from constants import Constants
from openpyxl import load_workbook

class ReportingModel:
    def __init__(self, forecast_output, pre_model_output, output_dir):
        self.forecast_output = forecast_output
        self.pre_model_output = pre_model_output
        self.output_dir = output_dir

    def report(self):
        pre_model = pd.read_csv(self.pre_model_output, low_memory=False,
                                parse_dates=['calendardate'])
        # pre_model = pre_model.sort_values(by=['ticker', 'calendardate'], ascending=[True, False])
        # pre_model = pre_model.drop_duplicates(subset='ticker', keep='first')
        forecast = pd.read_csv(self.forecast_output, low_memory=False, parse_dates=['calendardate'])
        forecast = forecast[['ticker', 'calendardate', 'probability', 'result']]

        # add the last price
        output = pd.merge(forecast,
                          pre_model,
                          on=['ticker', 'calendardate'])

        output = output[(output['probability'].notna()) &
                        (output['future_price'].isna())]
        output['invest'] = (((output['last_price'] - output['price']) / output[
            'price'] < Constants.get_target_price_appreciation()) &
                            (output['result'] == 1))

        start_columns = ['ticker', 'calendardate', 'invest', 'result', 'probability', 'sector', 'industry',
                         'scalemarketcap', 'scalerevenue',
                         'pb', 'pe', 'ps', 'de', 'roa', 'roe', 'roic',
                         'revenueusd_1yr_change', 'revenueusd_3yr_change', 'revenueusd_5yr_change',
                         'gross_margin_1yr_change', 'gross_margin_3yr_change', 'gross_margin_5yr_change',
                         'net_margin_1yr_change', 'net_margin_3yr_change',
                         'net_margin_5yr_change']

        other_columns = [col for col in list(output.columns) if col not in start_columns]
        columns = start_columns + other_columns
        output = output[columns]
        output = output.sort_values(by=['invest', 'probability'], ascending=[False, False])
        output.to_csv(os.path.join(self.output_dir, 'report.csv'), index=False)

        excel_path = os.path.join(self.output_dir, 'report.xlsx')
        output.to_excel(excel_path, index=False)

        # enable auto filters
        wb = load_workbook(excel_path)
        ws = wb.active
        ws.auto_filter.ref = ws.dimensions
        wb.save(excel_path)


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
