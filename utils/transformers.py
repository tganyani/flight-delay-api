from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TimeToDecimal(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in X.columns:
            X[col] = X[col].apply(self.hhmm_to_decimal)

        return X

    @staticmethod
    def hhmm_to_decimal(time):

        if pd.isna(time):
            return time

        time = int(time)

        hours = time // 100
        minutes = time % 100

        return hours + minutes / 60


class ToStringTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str)