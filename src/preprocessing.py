import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def fill_missing_dates(df: pd.DataFrame):

    def fill_na(data: pd.DataFrame):
        start = data.datetime.min()
        end = data.datetime.max()
        idx = pd.date_range(start=start, end=end, freq='1min')
        data.set_index('datetime', inplace=True)
        data = data.reindex(idx)

        data.timestamp = (data.index.astype(int) / 10 ** 9)
        data.kpi_id = data.kpi_id.unique()[0]

        data.fillna(method="ffill", inplace=True)  # ffill: propagate last valid observation forward to next valid
        return data.reset_index(drop=False).rename(columns={'index': 'datetime'})

    df = df.groupby('kpi_id').apply(fill_na).reset_index(drop=True)
    return df


def split_train_val_timeseries(df: pd.DataFrame, id='kpi_id', train_val_split=.95) -> (pd.DataFrame, pd.DataFrame):
    train = df.groupby(id).apply(lambda x: x.iloc[:round(x.shape[0] * train_val_split)]).reset_index(drop=True)
    val = df.groupby(id).apply(lambda x: x.iloc[round(x.shape[0] * train_val_split):]).reset_index(drop=True)
    return train, val


def train_scalers_on_timeseries(df: pd.DataFrame, id='kpi_id', col='value', scaler=StandardScaler):
    return df.groupby(id)[col].apply(lambda x: scaler().fit(x.to_numpy().reshape(-1, 1))).rename('scaler')


def scale_timeseries(df: pd.DataFrame, scalers: pd.Series, col='value', id='kpi_id'):
    df[f'{col}_scaled'] = df.groupby(id).apply(lambda x: pd.Series(scalers[x[id].iloc[0]].transform(x[col].to_numpy().reshape(-1, 1)).reshape(-1,))).reset_index(drop=True)
    return df
