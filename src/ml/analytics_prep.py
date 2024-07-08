import sys
import zipfile
import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import DateOffset


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


class EquityModel:
    def __init__(self, data_dir, analytics_dir, num_rows):
        if not os.path.exists(analytics_dir):
            print(f"Not directory {analytics_dir} found")
            sys.exit(-1)

        # Initialize the zip file path
        self.num_rows = num_rows
        self.equity_data = None
        self.analytics_dir = analytics_dir
        self.real_rates = None
        self.equity_prices = None
        self.ticker_data = None
        self.model_ready_data = None
        self.actions = None
        self.last_price = None
        self.data_dir = data_dir

    def read_rates(self):
        # read real rates and calculate trailing 12 month
        self.real_rates = pd.read_csv("market_data/real_rates.csv")
        self.real_rates['Model Output Date'] = pd.to_datetime(self.real_rates['Model Output Date'], format='%m/%d/%y',
                                                              errors='coerce')
        self.real_rates = self.real_rates.rename(columns={
            'Model Output Date': 'real-rate-model-date',
            'Real Rate 1-month': 'real-rate-1-month',
            'Real Rate 1-year': 'real-rate-1-year',
            'Real Rate 10-year ': 'real-rate-10-year',
        })
        self.real_rates['calendardate'] = self.real_rates['real-rate-model-date'] - pd.DateOffset(days=1)
        self.real_rates.set_index('calendardate', inplace=True)
        columns_to_average = ['real-rate-1-month', 'real-rate-1-year', 'real-rate-10-year']
        for column in columns_to_average:
            self.real_rates[f'{column}_ttm'] = self.real_rates[column].rolling(window=12).mean()

        self.real_rates = self.real_rates.dropna()

    def read_SF1(self, num_rows=None):
        self.equity_data = Utils.unzip(os.path.join(dir, "SF1.zip"), num_rows=num_rows)
        self.equity_data = self.equity_data[self.equity_data['dimension'] == "MRT"]
        self.equity_data['calendardate'] = pd.to_datetime(self.equity_data['calendardate'])
        self.equity_data['reportperiod'] = pd.to_datetime(self.equity_data['reportperiod'])

    def read_prices(self, num_rows=None):
        self.equity_prices = Utils.unzip(os.path.join(dir, "SEP.zip"), num_rows=num_rows)
        self.equity_prices['date'] = pd.to_datetime(self.equity_prices['date'])

    def read_actions(self, num_rows=None):
        self.actions = Utils.unzip(os.path.join(dir, "ACTIONS.zip"), num_rows=num_rows)
        self.actions['date'] = pd.to_datetime(self.actions['date'])

    def read_tickers(self, num_rows=None):
        self.ticker_data = Utils.unzip(os.path.join(dir, "TICKERS.zip"), num_rows=num_rows)

    def read_data(self):
        self.read_rates()
        self.read_SF1(self.num_rows)
        self.read_prices(self.num_rows)
        self.read_actions(self.num_rows)
        self.read_tickers(self.num_rows)

        self.merga_tables()
        self.prep_SF1()

    def merga_tables(self):
        sf1_ticker_meta = self.ticker_data[self.ticker_data['table'] == 'SF1']
        select_ticker_fields = sf1_ticker_meta[
            ['ticker', 'siccode', 'sicsector', 'sicindustry', 'famasector', 'famaindustry', 'sector', 'industry']]

        # get sector information
        self.equity_data = pd.merge(
            self.equity_data,
            select_ticker_fields,
            how='left',
            left_on=['ticker'],
            right_on=['ticker']
        )

        # get real rates
        self.equity_data = pd.merge(
            self.equity_data,
            self.real_rates,
            how='left',
            left_on=['calendardate'],
            right_on=['calendardate']
        )
        if len(self.equity_data['real-rate-1-month'].isna()) == 0:
            print("Real rates not found.")
            print(self.equity_data[self.equity_data['real-rate-1-month'].isna()].head())
            sys.exit(-1)

    def create_industry_proxies(self):
        # create the ratios
        self.equity_data['debt_debtc'] = ((self.equity_data['debt'] / self.equity_data['debtc'])
                                          .replace([np.inf, -np.inf], np.nan))
        self.equity_data['assets_assetsc'] = ((self.equity_data['assets'] / self.equity_data['assetsc'])
                                              .replace([np.inf, -np.inf], np.nan))
        self.equity_data['liabilities_liabilitiesc'] = (self.equity_data['liabilities'] / self.equity_data[
            'liabilitiesc']).replace([np.inf, -np.inf], np.nan)

        # compute the averages
        industry_average = self.equity_data[
            ['industry', 'sector', 'debt_debtc', 'assets_assetsc', 'liabilities_liabilitiesc']]

        mean_industry_average = industry_average.groupby(['industry', 'sector'], dropna=True).agg({
            'debt_debtc': 'mean',
            'assets_assetsc': 'mean',
            'liabilities_liabilitiesc': 'mean',
        }).reset_index()

        mean_industry_average = mean_industry_average.rename(columns={
            'debt_debtc': 'mean_debt_to_debtc',
            'assets_assetsc': 'mean_assets_to_assetsc',
            'liabilities_liabilitiesc': 'mean_liabilities_to_liabilitiesc',
        })

        self.equity_data.drop(columns=[
            'debt_debtc', 'assets_assetsc', 'liabilities_liabilitiesc',
        ], inplace=True)

        return mean_industry_average

    def replace_with_industry_average(self):
        mean_industry_average = self.create_industry_proxies()

        self.equity_data = self.equity_data.merge(mean_industry_average, on=['industry', 'sector'], how='left')
        self.equity_data['debtc'] = np.where(
            pd.isna(self.equity_data['debtc']),
            self.equity_data['debt'] / self.equity_data['mean_debt_to_debtc'],
            self.equity_data['debtc']
        )
        self.equity_data['assetsc'] = np.where(
            pd.isna(self.equity_data['assetsc']),
            self.equity_data['assets'] / self.equity_data['mean_assets_to_assetsc'],
            self.equity_data['assetsc']
        )
        self.equity_data['liabilitiesc'] = np.where(
            pd.isna(self.equity_data['liabilitiesc']),
            self.equity_data['liabilities'] / self.equity_data['mean_liabilities_to_liabilitiesc'],
            self.equity_data['liabilitiesc']
        )

        self.equity_data.drop(columns=[
            'mean_debt_to_debtc',
            'mean_assets_to_assetsc',
            'mean_liabilities_to_liabilitiesc',
        ], inplace=True)

        mask = self.equity_data['assetsnc'].isna() & self.equity_data['assetsc'].notna()
        self.equity_data.loc[mask, 'assetsnc'] = (self.equity_data.loc[mask, 'assets'] -
                                                  self.equity_data.loc[mask, 'assetsc'])

        # handle case with non-currentliab not available
        mask = self.equity_data['liabilitiesnc'].isna() & self.equity_data['liabilitiesc'].notna()
        self.equity_data.loc[mask, 'liabilitiesnc'] = (self.equity_data.loc[mask, 'liabilities'] -
                                                       self.equity_data.loc[mask, 'liabilitiesc'])

        mask = self.equity_data['debtnc'].isna() & self.equity_data['debtc'].notna()
        self.equity_data.loc[mask, 'debtnc'] = (self.equity_data.loc[mask, 'debt'] -
                                                self.equity_data.loc[mask, 'debtc'])

    def prep_SF1(self):
        # sort the values
        self.equity_data = self.equity_data.sort_values(by=['ticker', 'calendardate'])

        # handle case with currentassets not available
        mask = self.equity_data['assetsc'].isna()
        self.equity_data.loc[mask, 'assetsc'] = (self.equity_data.loc[mask, ['cashneq', 'investments', 'receivables']]
                                                 .sum(axis=1, skipna=True))

        # handle case with currentliab not available
        mask = self.equity_data['liabilitiesc'].isna()
        self.equity_data.loc[mask, 'liabilitiesc'] = self.equity_data.loc[mask, ['payables']].sum(axis=1, skipna=True)

        # handle case with intangibles is not available. Fill with last available value (forward fill)
        self.equity_data['intangibles'] = self.equity_data.groupby('ticker')['intangibles'].ffill()

        # handle cases where the current and non current asset,libailities and debt is missing

        # replace key values with industry averages
        self.replace_with_industry_average()

        self.equity_data['assetsusd'] = self.equity_data['assets'] * self.equity_data['fxusd']
        self.equity_data['assetsavgusd'] = self.equity_data['assetsavg'] * self.equity_data['fxusd']
        self.equity_data['liabilitiesusd'] = self.equity_data['liabilities'] * self.equity_data['fxusd']
        self.equity_data['liabilitiescusd'] = self.equity_data['liabilitiesc'] * self.equity_data['fxusd']
        self.equity_data['liabilitiesncusd'] = self.equity_data['liabilitiesnc'] * self.equity_data['fxusd']
        self.equity_data['debtncusd'] = self.equity_data['debtnc'] * self.equity_data['fxusd']
        self.equity_data['debtcusd'] = self.equity_data['debtc'] * self.equity_data['fxusd']
        self.equity_data['intangiblesusd'] = self.equity_data['intangibles'] * self.equity_data['fxusd']
        self.equity_data['assetscusd'] = self.equity_data['assetsc'] * self.equity_data['fxusd']
        self.equity_data['assetsncusd'] = self.equity_data['assetsnc'] * self.equity_data['fxusd']
        self.equity_data['fcfusd'] = self.equity_data['fcf'] * self.equity_data['fxusd']

    def prepare_model_data(self):

        # get the last price of the ticker
        self.last_price = self.equity_prices.loc[self.equity_prices.groupby('ticker')['date'].idxmax()]
        self.last_price.rename(columns={'closeadj': 'last_price'}, inplace=True)
        self.last_price.rename(columns={'date': 'last_price_date'}, inplace=True)

        # only work with the last date of the quarter
        print(f"Processing quarterly stock price")
        self.equity_prices['quarter_end'] = self.equity_prices['date'] + pd.offsets.QuarterEnd(0)
        self.equity_prices = self.equity_prices.sort_values(by='date').groupby(
            ['ticker', 'quarter_end']).last().reset_index()
        self.equity_prices = self.equity_prices[['ticker', 'quarter_end', 'closeadj']]
        self.equity_prices.rename(columns={'closeadj': 'quarter_end_price'}, inplace=True)
        print(f"Finished processing quarterly stock price")

        # pre-process SF 1
        self.equity_data['future_price_date'] = self.equity_data['calendardate'] + pd.DateOffset(years=1)

        # merge the 2
        self.equity_data = pd.merge(
            self.equity_data,
            self.equity_prices,
            how='left',
            left_on=['ticker', 'future_price_date'],
            right_on=['ticker', 'quarter_end']
        )
        self.equity_data.rename(columns={'quarter_end_price': 'future_price'}, inplace=True)
        self.equity_data.drop(columns=['quarter_end'], inplace=True)

        # add the last price
        self.equity_data = pd.merge(self.equity_data,
                                    self.last_price[['ticker', 'last_price_date', 'last_price']],
                                    how='left',
                                    left_on='ticker',
                                    right_on='ticker')

        # add the bank/liqd info
        liquidation = self.actions[self.actions['action'] == "bankruptcyliquidation"]
        self.equity_data = pd.merge(self.equity_data,
                                    liquidation[['ticker', 'date', 'action']],
                                    how='left',
                                    left_on='ticker',
                                    right_on='ticker')
        self.equity_data.rename(columns={'date': 'liq_date'}, inplace=True)

        # add other liquidation etc
        other_term = self.actions[self.actions['action'].isin(["regulatorydelisting", "voluntarydelisting"])]
        self.equity_data = pd.merge(self.equity_data,
                                    other_term[['ticker', 'date', 'action']],
                                    how='left',
                                    left_on='ticker',
                                    right_on='ticker')
        self.equity_data.rename(columns={'date': 'corp_end_date'}, inplace=True)

        # cleanup final dataset
        is_liquidation = self.equity_data['future_price'].isna() & (self.equity_data['liq_date'].notna())
        self.equity_data.loc[is_liquidation, 'future_price'] = 0
        self.equity_data['future_return'] = ((self.equity_data['future_price'] - self.equity_data['price']) /
                                      self.equity_data['price'])
        self.equity_data['future_return'] = self.equity_data['future_return'].replace([np.inf, -np.inf], np.nan)

        # calculate missing fields

        # for gross margin, we'll set it to zero if revenue is zero
        self.equity_data['grossmargin'] = np.where(
            (self.equity_data['grossmargin'].isna()),
            0,
            self.equity_data['grossmargin'])

        # price to sales
        self.equity_data['ps'] = np.where(
            (self.equity_data['ps'].isna()),
            self.equity_data['marketcap']/self.equity_data['revenueusd'],
            self.equity_data['ps'])

        # price to earnings
        self.equity_data['pe'] = np.where(
            (self.equity_data['pe'].isna()),
            self.equity_data['marketcap']/self.equity_data['netinccmnusd'],
            self.equity_data['pe'])

        self.equity_data = self.equity_data.replace([np.inf, -np.inf], np.nan)

        # metric trends
        self.equity_data.set_index(['ticker', 'calendardate'], inplace=True)
        self.equity_data['revenueusd_1yr_change'] = self.equity_data['revenueusd'].pct_change(periods=4, fill_method=None)
        self.equity_data['revenueusd_3yr_change'] = self.equity_data['revenueusd'].pct_change(periods=12, fill_method=None)
        self.equity_data['revenueusd_5yr_change'] = self.equity_data['revenueusd'].pct_change(periods=20, fill_method=None)

        self.equity_data['gross_margin_1yr_change'] = self.equity_data['grossmargin'].pct_change(periods=4, fill_method=None)
        self.equity_data['gross_margin_3yr_change'] = self.equity_data['grossmargin'].pct_change(periods=12, fill_method=None)
        self.equity_data['gross_margin_5yr_change'] = self.equity_data['grossmargin'].pct_change(periods=20, fill_method=None)

        self.equity_data['net_margin_1yr_change'] = self.equity_data['netmargin'].pct_change(periods=4, fill_method=None)
        self.equity_data['net_margin_3yr_change'] = self.equity_data['netmargin'].pct_change(periods=12, fill_method=None)
        self.equity_data['net_margin_5yr_change'] = self.equity_data['netmargin'].pct_change(periods=20, fill_method=None)

        self.equity_data.reset_index(inplace=True)
        self.equity_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # store the main data with some basic filters
        basic_mask = (self.equity_data['revenue'].notna() & self.equity_data['revenue'] != 0)
        self.equity_data = self.equity_data[basic_mask]

        liq_mask = self.equity_data['liq_date'].notna() & (self.equity_data['liq_date'] < self.equity_data['calendardate'])
        basic_data = self.equity_data[~liq_mask]
        basic_data.to_csv(os.path.join(self.analytics_dir, 'pre_model_ready.csv'), index=False)

        # key fields that are needed for model processing
        filtered_mask = (self.equity_data['future_price'].notna()) & \
                        (self.equity_data['revenue'].notna() & self.equity_data['revenue'] != 0) & \
                        (self.equity_data['price'].notna()) & \
                        (self.equity_data['future_return'].notna())

        filtered_data = self.equity_data[filtered_mask]

        # filtered_mask = filtered_data['liq_date'].notna() & (filtered_data['liq_date'] < filtered_data['calendardate'])
        # filtered_data = filtered_data[~filtered_mask]

        filtered_data = filtered_data[
            ['ticker',
             'calendardate',
             'reportperiod',
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
             'sicindustry',
             'sicsector',
             'sector',
             'industry',
             'future_return',
             'real-rate-1-month_ttm',
             'real-rate-1-year_ttm',
             'real-rate-10-year_ttm',
             ]]


        dropped_records = filtered_data[filtered_data.isna().any(axis=1)]
        dropped_records.to_csv(os.path.join(self.analytics_dir, 'dropped_records.csv'), index=False)

        self.model_ready_data = filtered_data.dropna()
        self.model_ready_data.to_csv(os.path.join(self.analytics_dir, 'model_ready.csv'), index=False)
        print(f"Done calculating data model ready data. Original data length: {len(self.equity_data)}")
        print(f"filtered: {len(filtered_data)} market_data:{len(self.model_ready_data)}")


if __name__ == "__main__":
    dir = sys.argv[1]
    analytics_dir = sys.argv[2]
    num_rows = None
    if len(sys.argv) > 3:
        num_rows = int(sys.argv[3])

    print(f"Running prep data_dir={dir} output_dir={analytics_dir} for num_rows:{num_rows}")
    model = EquityModel(dir, analytics_dir, num_rows=num_rows)
    model.read_data()
    model.prepare_model_data()
    print('Done')
