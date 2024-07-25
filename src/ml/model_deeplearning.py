import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from market_data import MarketData
from model_base import BaseModel


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


class DeepLearningModel(BaseModel):
    def __init__(self, data_file, model_path, train_mode=True, epochs=100, sequence_length=3):
        super().__init__(data_file, model_path)
        self.train_mode = train_mode
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.t = None

    def load_data(self, num_rows=None):
        super().load_data()

    def get_model_columns(self):
        model_columns = ['ticker', 'calendardate', 'reportingquarter', 'eps', 'bvps', 'dps', 'divyield', 'revenueusd',
                         'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd', 'liabilitiesusd',
                         'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd', 'debtcusd', 'debtncusd',
                         'intangiblesusd', 'fcfusd', 'marketcap', 'ps', 'pe', 'roe', 'roa', 'pb', 'de', 'netmargin',
                         'grossmargin', 'sector', 'industry']
        model_columns.extend(MarketData.get_indicators_ttm())
        model_columns.extend(['result'])

        return model_columns

    def get_large_numeric_columns(self):
        return [
            'eps', 'bvps', 'dps', 'divyield',
            'revenueusd', 'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd',
            'liabilitiesusd', 'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd',
            'debtcusd', 'debtncusd', 'intangiblesusd', 'fcfusd', 'marketcap',
            'ps', 'pe', 'roe', 'roa', 'pb', 'de', 'netmargin', 'grossmargin'
        ]

    def scale_data(self):
        model_data = self.data.copy()
        model_data = model_data.sort_values(by=['ticker', 'calendardate'])

        label_encoder_sector = LabelEncoder()
        if 'sector' in model_data.columns:
            model_data['sector'] = label_encoder_sector.fit_transform(model_data['sector'])

        label_encoder_industry = LabelEncoder()
        if 'industry' in model_data.columns:
            model_data['industry'] = label_encoder_industry.fit_transform(model_data['industry'])

        # use the columns if provided
        large_numeric_cols = self.get_large_numeric_columns()
        large_numeric_cols = list(set(model_data.columns).intersection(large_numeric_cols))

        # all these treated as standard columns
        market_data_columns = MarketData.get_indicators_ttm()
        market_data_columns = list(set(model_data.columns).intersection(market_data_columns))

        # scale financial data
        print(f"Scaling financial data")
        robust_scaler = RobustScaler()
        pt = PowerTransformer(method='yeo-johnson')

        if len(large_numeric_cols) > 0:
            model_data[large_numeric_cols] = robust_scaler.fit_transform(model_data[large_numeric_cols])
            model_data[large_numeric_cols] = pt.fit_transform(model_data[large_numeric_cols])

        # standard scaler
        scaler = StandardScaler()
        if len(market_data_columns) > 0:
            model_data[market_data_columns] = scaler.fit_transform(model_data[market_data_columns])
        print(f"Done scaling financial data")

        self.model_data = model_data

    @staticmethod
    def create_sequences_for_ticker(ticker, ticker_df, sequence_length):
        data = ticker_df.values
        sequences = []

        for i in range(len(ticker_df) - sequence_length + 1):
            sequence = data[i:i + sequence_length, :-1]
            target = data[i + sequence_length - 1, -1]
            sequences.append((ticker, sequence, target))

        return sequences

    def prep_data(self):

        # scale the data
        self.scale_data()

        if not self.train_mode:
            self.model_data = self.model_data.sort_values(by=['ticker', 'calendardate'], ascending=[True, False])
            self.model_data = self.model_data.groupby('ticker').filter(lambda x: len(x) >= self.sequence_length)
            self.model_data = self.model_data.groupby('ticker').head(3)

        self.model_data = self.model_data[self.get_model_columns()]

        # create ticker sequences
        data = self.model_data.sort_values(by=['ticker', 'calendardate'])

        print(f"Creating ticker sequences")
        all_sequences = []
        ticker_groups = data.groupby('ticker')

        print(f"Splitting into sequences")
        for ticker, ticker_df in ticker_groups:
            if len(ticker_df) >= self.sequence_length:
                ticker_df = ticker_df.drop(columns=['ticker', 'calendardate'])
                ticker_sequences = self.create_sequences_for_ticker(ticker, ticker_df, self.sequence_length)
                all_sequences.extend(ticker_sequences)

        print(f"Done creating ticker sequences")

        # Split sequences into inputs and targets
        t, x, y = zip(*all_sequences)
        t = np.array(t)
        x = np.array(x)
        y = np.array(y)

        # get column name
        columns = list(data.columns)
        for value in ['ticker', 'calendardate', 'result']:
            if value in columns:
                columns.remove(value)

        # Padding sequences to ensure they are of equal length
        # x_padded = pad_sequences(x, maxlen=self.sequence_length, dtype='float32', padding='post', truncating='post')
        self.t, self.x, self.y = t, x, y

    def get_input_shape(self):
        return self.sequence_length, self.x.shape[2]

    def build_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def train_model(self):
        model_exists = self.load_model()

        if not model_exists:
            print("Creating a new checkpoint")

        # Set up the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=self.model_path,
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
            self.x, self.y,
            epochs=self.epochs,
            validation_split=BaseModel.get_train_test_split(),
            batch_size=16,
            callbacks=[checkpoint_callback, early_stopping, plot_callback]
        )
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Ensure the final plot is shown

        return history

    def get_model_features(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def run_predictions(self):
        # get the original data
        output = self.data.copy()
        if 'result' in output.columns:
            output = output.drop(columns=['result'])
        output = output.sort_values(by=['ticker', 'calendardate'], ascending=[True, False])
        output = self.model_data.drop_duplicates(subset='ticker', keep='first')

        # get the model output
        probability = self.model.predict(self.x)

        assert len(probability) == len(self.t)
        assert len(output) == len(probability)

        results = pd.DataFrame(self.t, columns=['ticker'])
        results['probability'] = probability
        results['result'] = (probability >= 0.5).astype(int)

        # combine
        merged = pd.merge(output, results, on=['ticker'])
        return merged

    def get_accuracy(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from checkpoint: {self.model_path}")
            self.model = load_model(self.model_path)
            return True
        else:
            print(f"model file {self.model_path} does not exist")
            return False

    def save_model(self):
        print(f"Saving model to {self.model_path}")
        self.model.save(self.model_path)


class CnnModel(DeepLearningModel):
    def build_model(self):
        input_shape = self.get_input_shape()
        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()


class LstmModel(DeepLearningModel):
    def build_model(self):
        input_shape = self.get_input_shape()

        self.model = Sequential()
        self.model.add(InputLayer(shape=input_shape))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
