
import argparse
from alpaca.data import TimeFrame, TimeFrameUnit
from datetime import datetime, timezone
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
import yaml
import os
from ai_stock_forecasts.model.model.model_module import ModelModule
from ai_stock_forecasts.model.trading_algorithms.simple_x_days_ahead_buying import SimpleXDaysAheadBuying
import sys

from ai_stock_forecasts.model.data.training_data_module import TrainingDataModule


def local_rank() -> int:
    v = os.environ.get("LOCAL_RANK")
    return int(v) if v is not None else -1

def global_rank() -> int:
    v = os.environ.get("RANK")
    return int(v) if v is not None else -1


class Orchestration:
    def __init__(self, symbols: list[str], model_id: str, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f) or {}

        self.symbols = symbols
        self.model_id = model_id

        if 'configs' not in full_config:
            raise Exception(f'the config path: {config_path} passed is not valid')

        if self.model_id not in full_config['configs']:
            raise Exception(f'The config id: {self.model_id} passed is not defined in configurations')

        self.config = full_config['configs'][self.model_id]

        self.train_start = datetime.fromisoformat(self.config['train_start']).replace(tzinfo=timezone.utc)
        self.train_end = datetime.fromisoformat(self.config['train_end']).replace(tzinfo=timezone.utc)
        self.val_end = datetime.fromisoformat(self.config['val_end']).replace(tzinfo=timezone.utc)
        self.test_end = datetime.fromisoformat(self.config['test_end']).replace(tzinfo=timezone.utc)

        self.features: list[str] = self.config['features']

        self.batch_size: int = self.config['batch_size']
        self.num_workers: int = self.config['num_workers']

        if sys.platform == 'darwin':
            self.accelerator: str = 'mps'
        else:
            self.accelerator: str = self.config['accelerator']

        self.devices: int = self.config['devices']

        time_frame_amount: int = self.config['time_frame_amount']
        time_frame_unit: str = self.config['time_frame_unit']
        self.time_frame: TimeFrame = TimeFrame(time_frame_amount, TimeFrameUnit(time_frame_unit))

        self.max_lookback_period: int = self.config['max_lookback_period']
        self.max_prediction_length: int = self.config['max_prediction_length']

        self.is_df_cached = not local_rank() == 0 and global_rank() == 0 and self.devices > 1

        self.use_gpu = self.accelerator in ('gpu', 'cuda', 'mps')

        self.learning_rate: float = self.config['learning_rate']
        self.hidden_size: int = self.config['hidden_size']
        self.attention_head_size: int = self.config['attention_head_size']
        self.dropout: float = self.config['dropout']
        self.hidden_continuous_size: int = self.config['hidden_continuous_size']
        self.lstm_layers: int = self.config['lstm_layers']
        self.reduce_on_plateau_patience = self.config['reduce_on_plateau_patience']
        self.max_epochs: int = self.config['max_epochs']

        if 'fine_tuning_model_id' in self.config:
            self.fine_tuning_model_id = self.config['fine_tuning_model_id']
        else:
            self.fine_tuning_model_id = None


    def run_training(self):
        self.training_data_module = TrainingDataModule(self.symbols, self.features,
                                                       self.time_frame,
                                                       self.max_lookback_period, self.max_prediction_length, self.is_df_cached)

        self.training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        self.training_data_module.construct_train_and_validation_dataloaders(self.batch_size, self.num_workers, self.use_gpu)

        if self.config['devices'] > 1:
            self.training_data_module.cache_df()

        self.model_module = ModelModule()

        if self.fine_tuning_model_id:
            self._load_model(self.fine_tuning_model_id)

        if (not isinstance(self.training_data_module.training_dataset, TimeSeriesDataSet) or
           not isinstance(self.training_data_module.train_dataloader, DataLoader) or
           not isinstance(self.training_data_module.validation_dataloader, DataLoader)):
            raise Exception('something went wrong...')
        else:
            self.model_module.run_training(self.training_data_module.training_dataset, self.learning_rate, self.hidden_size,
                                           self.attention_head_size, self.dropout, self.hidden_continuous_size,
                                           self.lstm_layers, self.reduce_on_plateau_patience, self.max_epochs,
                                           self.accelerator, self.devices, self.training_data_module.train_dataloader,
                                           self.training_data_module.validation_dataloader)

    def run_batch_inference(self):
        self.training_data_module = TrainingDataModule(self.symbols, self.features,
                                                       self.time_frame,
                                                       self.max_lookback_period, self.max_prediction_length, self.is_df_cached)

        self.training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        self.training_data_module.construct_train_and_validation_dataloaders(self.batch_size, self.num_workers, self.use_gpu)

        self.training_data_module.construct_test_dataset(self.train_start, self.val_end, self.test_end)
        self.training_data_module.construct_test_dataloader(self.batch_size, self.num_workers, self.use_gpu)

        self.model_module = ModelModule()

        self._load_model()

        if (not isinstance(self.training_data_module.test_dataloader, DataLoader)):
            raise Exception('something went wrong...')
        else:
            self.model_module.run_batch_inference(self.training_data_module.test_dataloader, self.model_id, self.training_data_module.df)

    def run_evaluation(self):
        self.model_module = ModelModule()
        try:
            self.model_module.load_human_readable_predictions(self.model_id)
        except:
            raise Exception('You must run batch inference before attempting to run evaluation')

        self.trading_algorithm = SimpleXDaysAheadBuying(num_stocks_purchased=10, capital_gains_tax=0.35)

        self.trading_algorithm.simulate(self.model_module.predictionsDF)


    def _load_model(self, model_id=''):
        model_id = model_id if model_id != '' else self.model_id
        try:
            self.model_module.load_model_from_checkpoint(model_id, self.accelerator)
        except:
            if (not isinstance(self.training_data_module.training_dataset, TimeSeriesDataSet)):
                raise Exception('something went wrong...')
            else:
                self.model_module.load_model_from_checkpoint_and_data(model_id, self.accelerator, 
                                                                      self.training_data_module.training_dataset,
                                                                      self.learning_rate, self.hidden_size, 
                                                                      self.attention_head_size, self.dropout,
                                                                      self.hidden_continuous_size, self.lstm_layers, 
                                                                      self.reduce_on_plateau_patience)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbols_path', type=str, default='/Users/michael/Coding/AIForecasts/AIStockForecasts/src/ai_stock_forecasts/constants/symbols.txt')
    parser.add_argument('--config_path', type=str, default='/Users/michael/Coding/AIForecasts/AIStockForecasts/src/ai_stock_forecasts/constants/configs.yaml')
    parser.add_argument('--model_id', type=str, default='m1-simple-daily-1-with-time-features-fine-tuned')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.symbols_path, "r") as f:
        symbols = [line.strip() for line in f]

    # when running locally
    if os.environ.get("SM_MODEL_DIR") is None:
        os.environ['SM_MODEL_DIR'] = './'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'

    orc = Orchestration(symbols, args.model_id, args.config_path)

    #orc.run_training()
    orc.run_batch_inference()
    orc.run_evaluation()



