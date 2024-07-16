import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
import random
import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


class AnalyticsCommon:
    @staticmethod
    def get_sequence():
        return 3


    @staticmethod
    def get_model_columns():
        return [
            'ticker',
            'calendardate',
            'reportingquarter',
            'eps',
            'bvps',
            'dps',
            'divyield',
            'revenueusd',
            'netinccmnusd',
            'equityusd',
            'assetsusd',
            'debtusd',
            'cashnequsd',
            'liabilitiesusd',
            'liabilitiescusd',
            'liabilitiesncusd',
            'assetscusd',
            'assetsncusd',
            'debtcusd',
            'debtncusd',
            'intangiblesusd',
            'fcfusd',
            'marketcap',
            'ps',
            'pe',
            'roe',
            'roa',
            'pb',
            'de',
            'netmargin',
            'grossmargin',
            'sector',
            'industry',
            'price_crude_oil_ttm',
            'price_natural_gas_ttm',
            'price_gold_ttm',
            'price_copper_ttm',
            'price_iron_ttm',
            'price_steel_ttm',
            'price_wheat_ttm',
            'price_corn_ttm',
            'price_soybeans_ttm',
            'price_rice_ttm',
            'price_real_estate_US_ttm',
            'rate_treasury_30_yr_ttm',
            'rate_treasury_10_yr_ttm',
            'rate_treasury_1_yr_ttm',
            'rate_real_rate_10_yr_ttm',
            'rate_real_rate_1_yr_ttm',
            'rate_real_rate_1_mth_ttm',
            'result'
        ]

    @staticmethod
    def read_file(data_file, num_rows=None):
        data = None
        print(f"Reading file {data_file} and rows:{num_rows}")
        if num_rows == None:
            data = pd.read_csv(data_file)
        else:
            data = pd.read_csv(data_file, nrows=num_rows)

        print(f"Done reading file {data_file} and rows:{num_rows}")

        data['calendardate'] = pd.to_datetime(data['calendardate'])
        data['reportingquarter'] = data['calendardate'].dt.quarter
        data = data[AnalyticsCommon.get_model_columns()]

        return data

    @staticmethod
    def create_sequences_for_ticker(ticker, ticker_df, sequence_length):
        data = ticker_df.values
        sequences = []

        for i in range(len(ticker_df) - sequence_length + 1):
            sequence = data[i:i + sequence_length, :-1]
            target = data[i + sequence_length - 1, -1]  # Adjusted to get the target correctly
            sequences.append((ticker, sequence, target))

        return sequences

    @staticmethod
    def preprocess(data, sequence_length):
        data = data.sort_values(by=['ticker', 'calendardate'])

        label_encoder_sector = LabelEncoder()
        data['sector'] = label_encoder_sector.fit_transform(data['sector'])

        label_encoder_industry = LabelEncoder()
        data['industry'] = label_encoder_industry.fit_transform(data['industry'])

        large_numeric_cols = [
            'eps', 'bvps', 'dps', 'divyield',
            'revenueusd', 'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd',
            'liabilitiesusd', 'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd',
            'debtcusd', 'debtncusd', 'intangiblesusd', 'fcfusd', 'marketcap',
            'ps', 'pe', 'roe', 'roa', 'pb', 'de', 'netmargin', 'grossmargin'
        ]

        standard_numeric_cols = [
            'price_crude_oil_ttm',
            'price_natural_gas_ttm',
            'price_gold_ttm',
            'price_copper_ttm',
            'price_iron_ttm',
            'price_steel_ttm',
            'price_wheat_ttm',
            'price_corn_ttm',
            'price_soybeans_ttm',
            'price_rice_ttm',
            'price_real_estate_US_ttm',
            'rate_treasury_30_yr_ttm',
            'rate_treasury_10_yr_ttm',
            'rate_treasury_1_yr_ttm',
            'rate_real_rate_10_yr_ttm',
            'rate_real_rate_1_yr_ttm',
            'rate_real_rate_1_mth_ttm',
        ]

        # scale financial data
        print(f"Scaling financial data")
        robust_scaler = RobustScaler()
        data[large_numeric_cols] = robust_scaler.fit_transform(data[large_numeric_cols])

        pt = PowerTransformer(method='yeo-johnson')
        data[large_numeric_cols] = pt.fit_transform(data[large_numeric_cols])

        # standard scaler
        scaler = StandardScaler()
        data[standard_numeric_cols] = scaler.fit_transform(data[standard_numeric_cols])
        print(f"Done scaling financial data")

        print(f"Creating ticker sequences")
        all_sequences = []
        ticker_groups = data.groupby('ticker')

        print(f"Splitting into sequences")
        for ticker, ticker_df in ticker_groups:
            if len(ticker_df) >= sequence_length:
                ticker_df = ticker_df.drop(columns=['ticker', 'calendardate'])
                ticker_sequences = AnalyticsCommon.create_sequences_for_ticker(ticker, ticker_df, sequence_length)
                all_sequences.extend(ticker_sequences)

        print(f"Done creating ticker sequences")
        # Split sequences into inputs and targets
        t, x, y = zip(*all_sequences)
        t = np.array(t)
        x = np.array(x)
        y = np.array(y)

        # Padding sequences to ensure they are of equal length
        X_padded = pad_sequences(x, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')

        return t, X_padded, y

    def split_train_test(x, y, test_size=0.2):
        # Split into training and testing sets
        print(f"Creating test and train datasets")
        return train_test_split(x, y, test_size=test_size, random_state=42)
