import pandas as pd
import requests
import sys
from io import BytesIO
from functools import reduce


class AvgUtils:
    @staticmethod
    def get_ttm_data(df, columns_to_average):
        freq = pd.infer_freq(df.index)
        df = df.resample('ME').mean()

        for column in columns_to_average:
            df[f'{column}_ttm'] = df[column].rolling(window=12).mean()
        df.drop(columns=columns_to_average, inplace=True)
        df = df.dropna()
        return df


class MarketData:
    @staticmethod
    def get_indicators_ttm():
        return [key + '_ttm' for key in MarketData.get_indicators().keys()]

    @staticmethod
    def get_indicators():
        return {
            'price_crude_oil': 'DCOILWTICO',
            'price_natural_gas': 'PNGASUSUSDM',
            'price_gold': 'IQ12260',
            'price_copper': 'PCOPPUSDM',
            'price_iron': 'PIORECRUSDM',
            'price_steel': 'WPU1017',
            'price_wheat': 'PWHEAMTUSDM',
            'price_corn': 'PMAIZMTUSDM',
            'price_soybeans': 'PSOYBUSDM',
            'price_rice': 'PRICENPQUSDM',
            'price_real_estate_US': 'CSUSHPINSA',
            'rate_treasury_30_yr': 'DGS30',
            'rate_treasury_10_yr': 'DGS10',
            'rate_treasury_1_yr': 'DGS1',
            'rate_real_rate_1_mth': 'REAINTRATREARAT1MO',
            'rate_real_rate_10_yr': 'REAINTRATREARAT10Y',
            'rate_real_rate_1_yr': 'REAINTRATREARAT1YE',
            'jobs': 'PAYEMS',
            'unemployment_rate': 'UNRATE',
            'savings_rate': 'PSAVERT',
            'disposable_income': 'DSPIC96',
            'real_disposable_income': 'DSPIC96',
            'retail_sales': 'RSAFS',
            'industrial_production' : 'INDPRO',
            'cpi' : 'CPIAUCSL'
        }

    def __init__(self, api_key):
        self.api_key = api_key
        self.indicators = MarketData.get_indicators()

    @staticmethod
    def merge_dfs(left, right):
        return pd.merge(left, right, on='date')

    def get_market_data(self):
        df_list = []
        for data_type, series_id in self.indicators.items():
            df = FREDApi.get_data(api_key=self.api_key,
                                  series_id=series_id,
                                  add_ttm=True,
                                  column_names={
                                      'date': 'date',
                                      'value': data_type
                                  })
            print(f"Read {data_type} with {len(df)} rows")
            if df is None:
                print(f"The market data for {data_type} is missing.")
                sys.exit()
            df_list.append(df)

        merged_df = reduce(MarketData.merge_dfs, df_list)
        return merged_df


class FREDApi:
    @staticmethod
    def get_data(api_key, series_id, add_ttm=False, column_names=None):
        url = f'https://api.stlouisfed.org/fred/series/observations'
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json'
        }

        # Make the API request
        response = requests.get(url, params=params)
        data = response.json()

        # Check for errors
        if 'error_code' in data:
            raise ValueError(f"Error {data['error_code']}: {data['error_message']}")
        else:
            # Extract observations
            observations = data['observations']

            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df.drop(columns=['realtime_start', 'realtime_end'], inplace=True)

            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['value'] = df['value'].ffill()
            df.set_index('date', inplace=True)
            df = df.dropna()

            if column_names is not None:
                df.rename(columns=column_names, inplace=True)

            if add_ttm:
                column_to_avg = [column_names['value'] if column_names is not None else 'value']
                df = AvgUtils.get_ttm_data(df, column_to_avg)
            df.reset_index(inplace=True)
            return df
