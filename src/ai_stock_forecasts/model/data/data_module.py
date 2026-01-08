from abc import abstractmethod

from alpaca.data import TimeFrame, TimeFrameUnit
from pandas import DataFrame, Series
from pytorch_forecasting import NaNLabelEncoder

class DataModule:
    def __init__(self, symbols: list[str], features: list[str], time_frame: TimeFrame,
                 max_lookback_period: int, max_prediction_length: int):
        self.df: DataFrame = DataFrame()
        self.categorical_encoders = {}
        self.known_reals = ['timestamp']
        self.unknown_reals = ['close', 'high', 'hour_of_day', 'low', 'open', 'trade_count', 'volume', 'vwap']
        self.unknown_categoricals = []
        self.known_categoricals = ['year', 'month', 'day_of_month', 'day_of_week', 'hour_of_day', 'minute_of_day']
        
        self.symbols = symbols
        self.features = features
        self.time_frame = time_frame
        self.max_lookback_period = max_lookback_period
        self.max_prediction_length = max_prediction_length

        self._validate_base_inputs_are_valid()
        self._construct_categorical_encoders()
        self._construct_df()

    """
        pulls necessary data from s3 and constructs a raw dataframe with columns:
            symbol, timestamp, time_idx, feature_1, feature_2, ...

        the dataframe is later used to build the train / validation / test TimeSeriesDatasets and dataloaders
    """
    @abstractmethod
    def _construct_df(self):
        raise Exception('This is an abstract method')

    def _validate_base_inputs_are_valid(self):
        if len(self.symbols) == 0:
            raise Exception('Must pass more than 0 symbols')

        if len(self.features) == 0:
            raise Exception('Must pass more than 0 features')

        if self.time_frame.unit_value == TimeFrameUnit.Hour:
            raise Exception('Hour Unit time frame is not supported yet')

        invalid_features = [] 
        for feature in self.features:
            if (feature not in self.known_reals and
               feature not in self.unknown_reals and
               feature not in self.unknown_categoricals and
               feature not in self.known_categoricals):
                invalid_features.append(feature)

        if len(invalid_features) > 0:
            raise Exception(f"The following features {invalid_features} must be defined in the base DataModule class, otherwise we won't know how to categorize the feature.")

    def _construct_categorical_encoders(self):
        self.categorical_encoders['symbol'] = NaNLabelEncoder().fit(Series(self.symbols))
        for feature in self.features:
            if feature in self.known_categoricals or feature in self.unknown_categoricals:
                self.categorical_encoders[feature] = NaNLabelEncoder(add_nan=True)

    def _get_known_reals(self):
        base = {'symbol', 'time_idx'}
        return [f for f in self.features if f in self.known_reals and f not in base]

    def _get_unknown_reals(self):
        base = {'symbol', 'time_idx'}
        return [f for f in self.features if f in self.unknown_reals and f not in base]

    def _get_known_categoricals(self):
        base = {'symbol', 'time_idx'}
        return [f for f in self.features if f in self.known_categoricals and f not in base]

    def _get_unknown_categoricals(self):
        base = {'symbol', 'time_idx'}
        return [f for f in self.features if f in self.unknown_reals and f not in base]


