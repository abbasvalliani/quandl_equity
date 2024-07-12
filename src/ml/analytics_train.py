import random
import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from sklearn.inspection import permutation_importance
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from analytics_common import AnalyticsCommon
from constants import Constants

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
    def __init__(self, data_file, epochs, sequence_length):
        # Initialize the zip file path
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.data_file = data_file
        self.data = None
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def read_data(self, num_rows=None):
        self.data = AnalyticsCommon.read_file(self.data_file, num_rows)

    def prep_data(self):
        t, x, y = AnalyticsCommon.preprocess(data=self.data, sequence_length=self.sequence_length)
        self.x_train, self.x_test, self.y_train, self.y_test = AnalyticsCommon.split_train_test(x, y, test_size=0.2)

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
        # self.model.add(Dense(1))
        # self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])
        self.model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def load_checkpoint(self, checkpoint_path='model_checkpoint.keras'):
        if os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            self.model = load_model(checkpoint_path)
        else:
            print(f"model file {checkpoint_path} does not exist")
            sys.exit()

    def train_model(self, X_train, y_train, checkpoint_path='model_checkpoint.keras'):
        self.x_train = X_train
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
            self.x_train, self.y_train,
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
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
        print(f'Train Loss: {loss}, Train Accuracy: {accuracy}')
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
        return accuracy

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def train(self):
        self.read_data()
        self.prep_data()
        input_shape = (self.sequence_length, self.x_train.shape[2])
        self.build_model(input_shape)
        self.train_model(self.x_train, self.y_train)

    def feature_analysis(self):
        self.read_data()
        self.prep_data()
        self.load_checkpoint()

        baseline_accuracy = self.evaluate_model()

        feature_importances = []
        for i in range(self.x_train.shape[2]):
            X_test_permuted = self.x_test.copy()
            np.random.shuffle(X_test_permuted[:, :, i])
            permuted_accuracy = self.model.evaluate(X_test_permuted, self.y_test, verbose=0)[1]
            importance = baseline_accuracy - permuted_accuracy
            feature_importances.append(importance)

        # Create a DataFrame for better visualization
        feature_names = [f'feature_{i}' for i in range(self.x_train.shape[2])]
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        print(feature_importance_df)

    def test(self):
        self.read_data()
        self.prep_data()
        self.load_checkpoint()
        self.evaluate_model()


if __name__ == "__main__":
    file = sys.argv[1]
    arg_epochs = int(sys.argv[2])

    print(f"file:{file} epochs={arg_epochs}")
    model = AnalyticsModel(file, epochs=arg_epochs, sequence_length=Constants.get_sequence_length())
    model.train()
    # model.test()
    model.feature_analysis()
    print('Done')
