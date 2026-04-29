
from typing import Optional
from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.model.model_module import ModelModule
from ai_stock_forecasts.models.day_of_week import DayOfWeek
from ai_stock_forecasts.trading_algorithms.volatility_ranking import VolatilityRanking
import pandas as pd


def run_evaluation(self):
    if self.model_type == 'lgbm':
        self._lgbm_run_batch_inference(self)
    elif self.model_type == 'tft':
        self._tft_run_batch_inference(self)
    else:
        raise Exception(f'model type: {self.model_type} not supported')

def _tft_run_evaluation(self):
    model_module = ModelModule(self.loss)

    try:
        predictionsDF = model_module.load_human_readable_predictions(self.model_id)
    except:
        raise Exception('You must run batch inference before attempting to run evaluation')


    """ If our model is predicting something other than actual stock numbers
        we want to pull a real stock number instead and evaluate with that.

        For example, predicting open_log_return is a calculated field and we want to calculate how much money we would actually make. To do that we can either reverse engineer the feature or better yet, just pull open and use that instead. Which is what we are doing.
    """
    dummy_data_module = None
    filtered_df: Optional[pd.DataFrame] = None
    if self.target not in ['close', 'high', 'low', 'open']:
        dummy_data_module = TrainingDataModule(self.symbols, ['close', 'close_log_return'],
                                                   self.time_frame,
                                                   self.max_lookback_period,
                                                   self.max_prediction_length,
                                                   'close',
                                                   self.target_normalizer)

        predictionsDF = model_module.append_actuals_to_simple_predictions(predictionsDF, dummy_data_module.df)

        filtered_df = dummy_data_module.df.copy()
        filtered_df = filtered_df[filtered_df['timestamp'] <= self.val_end]


    # trading_algorithm = SimpleXDaysAheadBuying(interval_days=5, num_stocks_purchased=10, uncertainty_multiplier=0.000, filter_out_x_most_volatile=200, predicting_raw_num=self.target in ['close', 'high', 'low', 'open'], pivot_df=filtered_df, day_of_week=DayOfWeek.tuesday)
    trading_algorithm = VolatilityRanking(num_stocks_purchased=10, day_of_week=DayOfWeek.wednesday, volatility_importance=0.4)

    trading_algorithm.simulate(predictionsDF)



def _lgbm_run_batch_inference(self):
    pass

