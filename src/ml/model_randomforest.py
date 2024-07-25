import os
import sys

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from market_data import MarketData

from model_base import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, data_file, model_path, train_mode=True):
        super().__init__(data_file, model_path, train_mode)

    def get_model_columns(self):
        model_columns = ['ticker', 'calendardate', 'divyield', 'revenueusd',
                         'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd', 'liabilitiesusd',
                         'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd', 'debtcusd', 'debtncusd',
                         'intangiblesusd', 'fcfusd', 'marketcap', 'sector', 'industry'
                         ]

        # model_columns = ['ticker', 'calendardate', 'reportingquarter', 'eps', 'bvps', 'dps', 'divyield', 'revenueusd',
        #                  'netinccmnusd', 'equityusd', 'assetsusd', 'debtusd', 'cashnequsd', 'liabilitiesusd',
        #                  'liabilitiescusd', 'liabilitiesncusd', 'assetscusd', 'assetsncusd', 'debtcusd', 'debtncusd',
        #                  'intangiblesusd', 'fcfusd', 'marketcap', 'ps', 'pe', 'roe', 'roa', 'pb', 'de', 'netmargin',
        #                  'grossmargin', 'sector', 'industry']
        model_columns.extend(MarketData.get_indicators_ttm())
        model_columns.extend(['result'])

        return model_columns

    def load_data(self, num_rows=None):
        super().load_data()

        if not self.train_mode:
            self.data = self.data.sort_values(by=['ticker', 'calendardate'], ascending=[True, False])
            self.data = self.data.drop_duplicates(subset='ticker', keep='first')

    def prep_data(self):
        print("Preparing data")
        model_data = self.data.copy()
        model_data = model_data.sort_values(by=['ticker', 'calendardate'])

        label_encoder_sector = LabelEncoder()
        if 'sector' in model_data.columns:
            model_data['sector'] = label_encoder_sector.fit_transform(model_data['sector'])

        label_encoder_industry = LabelEncoder()
        if 'industry' in model_data.columns:
            model_data['industry'] = label_encoder_industry.fit_transform(model_data['industry'])

        self.model_data = model_data[self.get_model_columns()]

        if 'result' in self.model_data.columns:
            self.x = self.model_data.drop(columns=['result'])
            self.y = self.model_data['result']
        else:
            self.x = self.model_data
            self.y = None

        self.x = self.x.drop(columns=['ticker', 'calendardate'])
        # poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        # self.x = poly.fit_transform(self.x)

    def build_model(self):
        # self.model = XGBClassifier(
        #     n_estimators=200,
        #     max_depth=9,
        #     learning_rate=0.1,
        #     subsample=1,
        #     colsample_bytree=0.8,
        #     random_state=42)

        # 85%/80%
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=17,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1)

        # self.model = RandomForestClassifier(
        #     n_estimators=200,
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     max_depth=None,
        #     random_state=42)

    def train_model(self):
        self.split_train_test()

        print(f"Training model")
        self.model.fit(self.x_train, self.y_train)

        train_accuracy = self.model.score(self.x_train, self.y_train)
        test_accuracy = self.model.score(self.x_test, self.y_test)

        print(f"Training Accuracy: {train_accuracy}")
        print(f"Testing Accuracy: {test_accuracy}")

        model_features = self.get_model_features()
        print(f"Model Features: {model_features}")

        print("Calculating cross validation accuracy")
        scores = cross_val_score(self.model, self.x_train, self.y_train, cv=5)
        print(f"Cross-Validation Accuracy Scores: {scores}")
        print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

        print("Now training on the entire dataset")
        train_accuracy = self.model.score(self.x, self.y)

        print(f"Training Accuracy Final: {train_accuracy}")
        print(f"Model Features Final: {self.get_model_features()}")

    def get_model_features(self):
        feature_columns = self.get_feature_columns()
        importance = self.model.feature_importances_

        # Create a DataFrame for visualization
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values(by='importance', ascending=False)

        return feature_importance

    def run_predictions(self):
        output = self.data.copy()
        output = output.sort_values(by=['ticker', 'calendardate'])
        probabilities = self.model.predict_proba(self.x)
        probabilities_binary = self.model.predict(self.x)
        probability = probabilities[:, 1]
        assert len(output) == len(probability)

        output['probability'] = probability
        output['result'] = probabilities_binary
        return output

    def get_accuracy(self):
        baseline_accuracy = self.model.evaluate(self.x, self.y, verbose=0)[1]
        print(f"Baseline accuracy of the results {baseline_accuracy}")

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def save_model(self):
        # save the model
        joblib.dump(self.model, self.model_path)
        print(f'Model saved to {self.model_path}')

        # save the features
        model_features = self.get_model_features()
        directory = os.path.dirname(self.model_path)
        model_features.to_csv(os.path.join(directory, 'random_forest_features.csv'), index=False)
