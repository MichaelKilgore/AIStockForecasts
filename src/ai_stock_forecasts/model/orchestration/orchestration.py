
import argparse
from alpaca.data import TimeFrame, TimeFrameUnit
from datetime import datetime, timezone
from pytorch_forecasting import TimeSeriesDataSet
import torch
from torch.utils.data import DataLoader
import yaml
import os
from ai_stock_forecasts.dynamodb.dynamodb_util import DynamoDBUtil
from ai_stock_forecasts.model.data.inference_data_module import InferenceDataModule
from ai_stock_forecasts.model.model.model_module import ModelModule
from ai_stock_forecasts.model.trading_algorithms.base_trading_module import BaseTradingModule
from ai_stock_forecasts.model.trading_algorithms.simple_x_days_ahead_buying import SimpleXDaysAheadBuying
import sys

from ai_stock_forecasts.model.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.models.order import Order, OrderItem

from alpaca.trading.enums import OrderSide

from ai_stock_forecasts.ordering.order_util import OrderUtil
from ai_stock_forecasts.s3.s3_util import S3ParquetUtil

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

        self.db_util = DynamoDBUtil()
        self.order_util = OrderUtil()
        self.s3_util = S3ParquetUtil()

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
            self._load_model(self.fine_tuning_model_id, modify_dropout=True)

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

        if sys.platform == 'darwin':
            self.model_module.upload_checkpoints_to_s3(self.model_id)

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

        # self.trading_algorithm = SimpleXDaysAheadBuying(interval_days=2, num_stocks_purchased=10, capital_gains_tax=0.35, uncertainty_multiplier=0.1, dont_buy_negative_stocks=True)

        # self.trading_algorithm.simulate(self.model_module.predictionsDF)

        results = []
        for i in list([1,2]):
            for j in list([10,25,50]):
                for k in list([0.1, 0.2, 0.3, 0.4, 0.5]):
                    self.trading_algorithm = SimpleXDaysAheadBuying(interval_days=i, num_stocks_purchased=j, capital_gains_tax=0.35, uncertainty_multiplier=k, dont_buy_negative_stocks=True)

                    results.append(([i,j,k], self.trading_algorithm.simulate(self.model_module.predictionsDF)))
                    print(results[-1])
        print(results)

    def explain_model(self):
        self.training_data_module = TrainingDataModule(self.symbols, self.features,
                                                       self.time_frame,
                                                       self.max_lookback_period, self.max_prediction_length, self.is_df_cached)

        self.training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        self.training_data_module.construct_train_and_validation_dataloaders(self.batch_size, self.num_workers, self.use_gpu)

        self.training_data_module.construct_test_dataset(self.train_start, self.val_end, self.test_end)
        self.training_data_module.construct_test_dataloader(self.batch_size, self.num_workers, self.use_gpu)

        self.model_module = ModelModule()

        self._load_model()

        if (not isinstance(self.training_data_module.train_dataloader, DataLoader)):
            raise Exception('something went wrong...')
        else:
            self.model_module.interpret_predictions(self.training_data_module.train_dataloader)

    def run_inference(self):
        self.inference_data_module = InferenceDataModule(self.symbols, self.features, self.time_frame, 
                                                         self.max_lookback_period, self.max_prediction_length)

        self.model_module = ModelModule()

        # TODO: Sometimes when models are trained on different hardware, loading in the model this way doesn't work.
        self.model_module.load_model_from_checkpoint(self.model_id, self.accelerator)
 
        self.inference_data_module.construct_inference_dataset(self.model_module.model.hparams["dataset_parameters"])
        self.inference_data_module.construct_inference_dataloader(self.batch_size, self.num_workers, self.use_gpu)

        self.model_module.run_single_day_inference(self.inference_data_module.inference_dataloader, self.inference_data_module.df)

        self.trading_algorithm = SimpleXDaysAheadBuying(interval_days=2, num_stocks_purchased=50, capital_gains_tax=0.35, uncertainty_multiplier=0.3, dont_buy_negative_stocks=True)

        stocks = self.trading_algorithm.generate_buy_list(self.model_module.predictionsDF)

        print(stocks)

    def execute_buy(self, testing: bool=False):
        if not testing and not self.order_util.is_stock_market_open():
            print(f'Stock market not open right now, returning...')
            return

        # get latest order details
        latest_order = self.db_util.get_latest_order(self.model_id)
        money_to_invest = latest_order.total_money_invested if latest_order != None else 25000

        self.inference_data_module = InferenceDataModule(self.symbols, self.features, self.time_frame, 
                                                 self.max_lookback_period, self.max_prediction_length)

        # determine trading strategy
        self._init_trading_strategy() 

        # determine if we have waited long enough (interval_days)
        if not testing and (latest_order is not None and not self.inference_data_module.is_it_time_to_order_again(latest_order.order_timestamp, self.interval_days)):
            print(f'Based on the interval days for this trading strategy: {self.interval_days}, its too soon to execute a trade, returning early...')
            return

        # sell previous order
        new_money_left = money_to_invest
        if latest_order is not None:
            order_items = [
                OrderItem(r.symbol, round(r.quantity, 2), OrderSide.SELL)
                for r in latest_order.order_items
            ]
            sell_order = Order(self.model_id, datetime.now(), money_to_invest, order_items=order_items)
            new_money_left = self.inference_data_module.update_money_to_invest(sell_order)
            sell_order.total_money_invested = round(float(new_money_left), 2)
            self.db_util.upload_order(sell_order)
            self.order_util.close_all_positions()

        # execute trading strategy
        self.model_module = ModelModule()

        with self.s3_util.load_best_model_checkpoint(self.model_id) as ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            hp = ckpt["hyper_parameters"]
            params = hp["dataset_parameters"]
 
        self.inference_data_module.construct_inference_dataset(params)
        self.inference_data_module.construct_inference_dataloader(self.batch_size, self.num_workers, self.use_gpu)


        self.model_module.load_model_from_checkpoint_and_data(self.model_id, self.accelerator, 
                                                              self.inference_data_module.inference_dataset, self.learning_rate,
                                                              self.hidden_size, self.attention_head_size,
                                                              self.dropout, self.hidden_continuous_size,
                                                              self.lstm_layers, self.reduce_on_plateau_patience)


        self.model_module.run_single_day_inference(self.inference_data_module.inference_dataloader, self.inference_data_module.df)

        top_x = self.trading_algorithm.generate_buy_list(self.model_module.predictionsDF)

        money_to_invest_in_each = float(new_money_left) / len(top_x)

        top_x['quantity'] = money_to_invest_in_each / top_x['current_y']

        order_items = [
            OrderItem(r.symbol, round(r.quantity, 2), OrderSide.BUY)
            for r in top_x.itertuples(index=False)
        ]

        # upload orders to db
        order = Order(self.model_id, datetime.now(), round(new_money_left, 2), order_items=order_items)
        self.db_util.upload_order(order)

        # execute orders in alpaca
        self.order_util.place_order(order)


    def _init_trading_strategy(self) -> BaseTradingModule:
        strat = self.config['preferred_trading_strategy']

        performance_test = self.config[strat]['_test_strategy'] 

        self.interval_days = self.config[strat]['_interval_days']

        if performance_test == 'SimpleXDaysAheadBuying':
            self.trading_algorithm = SimpleXDaysAheadBuying(
                                        interval_days=self.interval_days, 
                                        num_stocks_purchased=self.config[strat]['_num_stocks_purchased'], 
                                        capital_gains_tax=self.config[strat]['_capital_gains_tax'],
                                        uncertainty_multiplier=self.config[strat]['_uncertainty_multiplier'],
                                        compound_money=self.config[strat]['_compound_money'],
                                        dont_buy_negative_stocks=self.config[strat]['_dont_buy_negative_stocks'])
        else:
            raise Exception('The trading strategy specified is not supported')

    def _load_model(self, model_id='', modify_dropout=False):
        model_id = model_id if model_id != '' else self.model_id
        if not modify_dropout:
            try:
                self.model_module.load_model_from_checkpoint(model_id, self.accelerator)
            except:
                dataset = self.training_data_module.training_dataset
 
                if (not isinstance(dataset, TimeSeriesDataSet)):
                    raise Exception('something went wrong...')
                else:
                    self.model_module.load_model_from_checkpoint_and_data(model_id, self.accelerator, 
                                                                          dataset, self.learning_rate,
                                                                          self.hidden_size, self.attention_head_size,
                                                                          self.dropout, self.hidden_continuous_size,
                                                                          self.lstm_layers, self.reduce_on_plateau_patience)
        else:
            dataset = self.training_data_module.training_dataset
 
            if (not isinstance(dataset, TimeSeriesDataSet)):
                raise Exception('something went wrong...')
            else:
                self.model_module.load_model_from_checkpoint_and_data(model_id, self.accelerator, 
                                                                      dataset, self.learning_rate,
                                                                      self.hidden_size, self.attention_head_size,
                                                                      self.dropout, self.hidden_continuous_size,
                                                                      self.lstm_layers, self.reduce_on_plateau_patience)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbols_path', type=str, default='/Users/michael/Coding/AIForecasts/AIStockForecasts/src/ai_stock_forecasts/constants/symbols.txt')
    parser.add_argument('--config_path', type=str, default='/Users/michael/Coding/AIForecasts/AIStockForecasts/src/ai_stock_forecasts/constants/configs.yaml')
    parser.add_argument('--model_id', type=str, default='m1-medium-high-with-less-features-and-earnings-calendar-features')
    # 0 = False, 1 = True
    parser.add_argument('--run_training', type=bool, default=0)
    parser.add_argument('--run_batch_inference', type=bool, default=0)
    parser.add_argument('--run_evaluation', type=bool, default=0)
    parser.add_argument('--explain_model', type=bool, default=0)
    parser.add_argument('--run_inference', type=bool, default=0)
    parser.add_argument('--execute_buy', type=bool, default=1)

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

    if args.run_training:
        orc.run_training()
    if args.run_batch_inference:
        orc.run_batch_inference()
    if args.run_evaluation:
        orc.run_evaluation()
    if args.explain_model:
        orc.explain_model()
    if args.run_inference:
        orc.run_inference()
    if args.execute_buy:
        orc.execute_buy(True)



