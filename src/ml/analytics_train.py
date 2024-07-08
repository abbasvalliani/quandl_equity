import random
import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


class PlotLearning(Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Turn on interactive mode

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.ax.clear()
        self.ax.plot(self.epochs, self.train_losses, label='Train Loss')
        self.ax.plot(self.epochs, self.val_losses, label='Validation Loss')
        self.ax.legend()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Validation Loss over Epochs')
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot


class AnalyticsModel:
    def __init__(self, data_file, epochs, sequence_length=3, max_tickers=None):
        # Initialize the zip file path
        self.epochs = epochs
        self.max_tickers = max_tickers
        self.sequence_length = sequence_length
        self.data_file = data_file
        self.data = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def read_data(self, num_rows=None):
        print(f"Reading file {self.data_file} and rows:{num_rows}")
        if num_rows == None:
            self.data = pd.read_csv(self.data_file)
        else:
            self.data = pd.read_csv(self.data_file, nrows=num_rows)

        print(f"Done reading file {self.data_file} and rows:{num_rows}")

        self.data['calendardate'] = pd.to_datetime(self.data['calendardate'])
        self.data['reportingquarter'] = self.data['calendardate'].dt.quarter
        self.data = self.data[[
            'ticker',
            'calendardate',
            'reportingquarter',
            'eps', 'bvps',
            'dps', 'divyield',
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
            'real-rate-1-month_ttm', 'real-rate-1-year_ttm', 'real-rate-10-year_ttm',
            'future_return'
        ]]

    def create_sequences_for_ticker(self, ticker_df, sequence_length):
        ticker_df = ticker_df.drop(columns=['ticker', 'calendardate'])
        data = ticker_df.values
        sequences = []

        for i in range(len(ticker_df) - sequence_length):
            sequence = data[i:i + sequence_length, :-1]
            target = data[i + sequence_length, -1]
            sequences.append((sequence, target))

        return sequences

    def process_ticker_group(self, group):
        return self.create_sequences_for_ticker(group, self.sequence_length)

    def prep_data(self):
        self.data = self.data.sort_values(by=['ticker', 'calendardate'])

        label_encoder_sector = LabelEncoder()
        self.data['sector'] = label_encoder_sector.fit_transform(self.data['sector'])

        label_encoder_industry = LabelEncoder()
        self.data['industry'] = label_encoder_industry.fit_transform(self.data['industry'])

        large_numeric_cols = [
            'eps', 'bvps','dps','divyield',
            'revenueusd', 'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd',
            'liabilitiesusd', 'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd',
            'debtcusd', 'debtncusd', 'intangiblesusd', 'fcfusd', 'marketcap',
            'ps', 'pe', 'roe', 'roa', 'pb', 'de', 'netmargin', 'grossmargin',
            'future_return'
        ]

        standard_numeric_cols = [
            'real-rate-1-month_ttm', 'real-rate-1-year_ttm', 'real-rate-10-year_ttm',
        ]

        # scale financial data
        print(f"Scaling financial data")
        robust_scaler = RobustScaler()
        self.data[large_numeric_cols] = robust_scaler.fit_transform(self.data[large_numeric_cols])

        pt = PowerTransformer(method='yeo-johnson')
        self.data[large_numeric_cols] = pt.fit_transform(self.data[large_numeric_cols])

        # standard scaler
        scaler = StandardScaler()
        self.data[standard_numeric_cols] = scaler.fit_transform(self.data[standard_numeric_cols])
        print(f"Done scaling financial data")

        print(f"Creating ticker sequences")
        all_sequences = []
        ticker_groups = self.data.groupby('ticker')

        print(f"Splitting into sequences")
        for ticker, ticker_df in ticker_groups:
            if len(ticker_df) >= self.sequence_length:
                ticker_sequences = self.create_sequences_for_ticker(ticker_df, self.sequence_length)
                all_sequences.extend(ticker_sequences)

        print(f"Done creating ticker sequences")

        if self.max_tickers is not None:
            all_sequences = random.sample(all_sequences, self.max_tickers)

        # print(all_sequences[0])

        # Split sequences into inputs and targets
        X, y = zip(*all_sequences)
        X = np.array(X)
        y = np.array(y)

        # Padding sequences to ensure they are of equal length
        X_padded = pad_sequences(X, maxlen=self.sequence_length, dtype='float32', padding='post', truncating='post')

        # Split into training and testing sets
        print(f"Creating test and train datasets")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_padded, y, test_size=0.2,
                                                                                random_state=42)

    def build_model(self, input_shape):
        print(f"Building model")

        self.model = Sequential()
        self.model.add(InputLayer(shape=input_shape))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def load_checkpoint(self, checkpoint_path='model_checkpoint.keras'):
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            self.model = load_model(checkpoint_path)
        else:
            print(f"model file {checkpoint_path} does not exist")
            sys.exit()

    def train_model(self, X_train, y_train, checkpoint_path='model_checkpoint.keras'):
        self.X_train = X_train
        self.y_train = y_train

        # Load from checkpoint if it exists
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            self.model = load_model(checkpoint_path)
        else:
            print("No checkpoint found, training from scratch")

        # Set up the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_freq='epoch',
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # Set up EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        plot_callback = PlotLearning()
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_split=0.2,
            batch_size=16,
            callbacks=[checkpoint_callback, early_stopping, plot_callback]
            # callbacks=[checkpoint_callback, early_stopping, plot_callback]
        )
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Ensure the final plot is shown

        return history

    def evaluate_model(self):
        print(f"Evaluating model")
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Loss: {loss}')

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def train(self):
        self.read_data()
        self.prep_data()
        input_shape = (self.sequence_length, self.X_train.shape[2])
        self.build_model(input_shape)
        self.train_model(self.X_train, self.y_train)

    def test(self):
        self.read_data()
        self.prep_data()
        self.load_checkpoint()
        self.evaluate_model()


if __name__ == "__main__":
    file = sys.argv[1]
    arg_epochs = int(sys.argv[2])
    arg_max_tickers = None

    if len(sys.argv) > 3:
        arg_max_tickers = int(sys.argv[3])

    print(f"file:{file} epochs={arg_epochs} max_tickers={arg_max_tickers}")
    model = AnalyticsModel(file, epochs=arg_epochs, sequence_length=3, max_tickers=arg_max_tickers)
    model.train()
    # model.test()
    print('Done')
