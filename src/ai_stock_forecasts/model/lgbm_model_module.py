import logging
from datetime import datetime
from typing import Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import DataFrame


_NON_FEATURE_COLS = {'symbol', 'timestamp', 'time_idx'}


def _to_utc_naive(series: pd.Series) -> pd.Series:
    if hasattr(series.dt, 'tz') and series.dt.tz is not None:
        return series.dt.tz_convert('UTC').dt.tz_localize(None)
    return series


def _dt_to_utc_naive(dt: Union[datetime, pd.Timestamp, str]) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tz is not None:
        return ts.tz_convert('UTC').tz_localize(None)
    return ts

'''
Notes:
    - Calculates y at runtime (profit / loss) in run_training, its not a feature stored in s3
    - LGBM is meant to be run on cpu, so it does. TODO: In future I could add support for GPU training apparently.
'''
class LgbmModelModule:
    def __init__(self) -> None:
        self.model: lgb.LGBMClassifier = None
        self.feature_cols: list[str] = []
        self.target: str = ''
        self.prediction_horizon: int = 0

    def run_training(self, df: DataFrame, train_start: datetime, train_end: datetime,
                     val_end: Optional[datetime], target: str,
                     prediction_horizon: int = 7,
                     n_estimators: int = 500, num_leaves: int = 63,
                     learning_rate: float = 0.05,
                     min_child_samples: int = 20) -> None:
        self.feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
        self.target = target
        self.prediction_horizon = prediction_horizon

        ts = _to_utc_naive(df['timestamp'])
        future = df.groupby('symbol')[target].shift(-prediction_horizon)
        current = df[target]

        y = (future > current).astype(int)
        valid = future.notna() & current.notna()

        X_train, y_train = self._build_train_xy(df, ts, y, valid, train_start, train_end)

        has_val = val_end is not None
        if has_val:
            X_val, y_val = self._build_val_xy(df, ts, y, valid, train_end, val_end)

        self.model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            verbose=-1,
        )

        if has_val:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            val_pred = self.model.predict(X_val)
            val_acc = float((val_pred == y_val.values).mean())
            base_rate = float(y_val.mean())
            logging.info(
                f'val accuracy: {val_acc:.4f} '
                f'(profit base rate: {base_rate:.4f}, n={len(y_val)})'
            )
        else:
            self.model.fit(X_train, y_train)

        self._log_feature_importance()



    def run_batch_inference(self, df: DataFrame, model_id: str,
                            save_predictions: bool = True) -> DataFrame:
        pass

    def run_single_day_inference(self, df: DataFrame) -> DataFrame:
        pass

    def upload_model(self, model_id: str) -> None:
        pass

    def load_model(self, model_id: str) -> None:
        pass

    def load_human_readable_predictions(self, model_id: str) -> DataFrame:
        pass

    def append_actuals_to_simple_predictions(self, predictionsDF: DataFrame,
                                             df: DataFrame) -> DataFrame:
        pass

    def plot_mape_by_symbol(self, predictionsDF: DataFrame) -> None:
        pass

    def _build_train_xy(self, df: DataFrame, ts: pd.Series, y: pd.Series,
                        valid: pd.Series, train_start: datetime,
                        train_end: datetime) -> tuple[DataFrame, pd.Series]:
        train_start = _dt_to_utc_naive(train_start)
        train_end = _dt_to_utc_naive(train_end)
        mask = (ts >= train_start) & (ts <= train_end) & valid
        X_train = df.loc[mask, self.feature_cols].astype(float)
        y_train = y.loc[mask]
        return X_train, y_train

    def _build_val_xy(self, df: DataFrame, ts: pd.Series, y: pd.Series,
                      valid: pd.Series, train_end: datetime,
                      val_end: datetime) -> tuple[DataFrame, pd.Series]:
        train_end = _dt_to_utc_naive(train_end)
        val_end = _dt_to_utc_naive(val_end)
        mask = (ts > train_end) & (ts <= val_end) & valid
        X_val = df.loc[mask, self.feature_cols].astype(float)
        y_val = y.loc[mask]
        return X_val, y_val

    def _log_feature_importance(self) -> None:
        importance = self.model.booster_.feature_importance(importance_type='gain')
        gains = sorted(zip(self.feature_cols, importance),
                       key=lambda x: x[1], reverse=True)[:20]
        logging.info('Top 20 features by gain:')
        for feat, gain in gains:
            logging.info(f'  {feat}: {gain:.2f}')
