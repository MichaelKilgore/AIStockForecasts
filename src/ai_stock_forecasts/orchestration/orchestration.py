
import argparse
from typing import Optional
from alpaca.data import TimeFrame, TimeFrameUnit
from datetime import datetime, timezone
from pytorch_forecasting import QuantileLoss, TimeSeriesDataSet
import torch
import yaml
import os
from ai_stock_forecasts.losses.weighted_quantile_loss import WeightedQuantileLoss
from ai_stock_forecasts.models.day_of_week import DayOfWeek
from ai_stock_forecasts.trading_algorithms.volatility_ranking import VolatilityRanking
from ai_stock_forecasts.utils.date_util import get_prev_market_open_day
from ai_stock_forecasts.utils.dynamodb_util import DynamoDBUtil
from ai_stock_forecasts.data.inference_data_module import InferenceDataModule
from ai_stock_forecasts.model.model_module import ModelModule
from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule
from ai_stock_forecasts.trading_algorithms.simple_x_days_ahead_buying import SimpleXDaysAheadBuying
import sys

from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.models.order import Order, OrderItem

from alpaca.trading.enums import OrderSide

from ai_stock_forecasts.ordering.order_util import OrderUtil
from ai_stock_forecasts.utils.s3_util import S3ParquetUtil

import pandas as pd
import math

import logging

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

        self.quantiles: list[float] = self.config['quantiles'] if 'quantiles' in self.config else [0.3, 0.5, 0.7]

        if 'loss' not in self.config:
            self.loss = QuantileLoss(quantiles=self.quantiles)
        elif self.config['loss'] == 'WeightedQuantileLoss':
            self.loss = WeightedQuantileLoss(quantiles=self.quantiles)

        if 'gradient_clip_val' not in self.config:
            self.gradient_clip_val = None
        else:
            self.gradient_clip_val = self.config['gradient_clip_val']

        if 'target' in self.config:
            self.target = self.config['target']
        else:
            self.target = 'open'

        if 'target_normalizer' in self.config:
            self.target_normalizer = self.config['target_normalizer']
        else:
            self.target_normalizer = 'auto'

        self.is_large = False
        if 'is_large' in self.config:
            self.is_large = self.config['is_large']

    def run_training(self):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                   self.time_frame,
                                                   self.max_lookback_period,
                                                   self.max_prediction_length,
                                                   self.target,
                                                   self.target_normalizer)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, val_dataloader = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        if self.config['devices'] > 1:
            training_data_module.cache_df()

        model_module = ModelModule(self.loss)

        if self.fine_tuning_model_id:
            self._load_model(model_module, train_dataset, self.fine_tuning_model_id, modify_dropout=True, load_last_ckpt=True)

        model_module.run_training(train_dataset, self.learning_rate, self.hidden_size,
                                   self.attention_head_size, self.dropout, self.hidden_continuous_size,
                                   self.lstm_layers, self.reduce_on_plateau_patience, self.max_epochs,
                                   self.accelerator, self.devices, train_dataloader,
                                   val_dataloader, self.gradient_clip_val)

        model_module.upload_checkpoints_to_s3(self.model_id)

    def run_batch_inference(self, save_predictions=True, load_last_ckpt=False):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                       self.time_frame,
                                                       self.max_lookback_period,
                                                       self.max_prediction_length,
                                                       self.target,
                                                       self.target_normalizer)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, val_dataloader = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        '''
        TODO: For whatever reason when you break up the data into multiple dataloaders, predictions can be slightly different, not sure why though.
        # if self.is_large:
        #     test_datasets = training_data_module.construct_test_datasets(train_dataset, self.train_start, self.val_end, self.test_end, 10, self.max_prediction_length)
        # else:

        Also I recently upgraded my systems total ram to 64gb so skipping breaking up the datasets unless I find I need to again in the future.
        '''
        test_datasets = training_data_module.construct_test_datasets(train_dataset, self.train_start, self.val_end, self.test_end)

        test_dataloaders = training_data_module.construct_test_dataloaders(test_datasets, self.batch_size, self.num_workers, self.use_gpu)

        model_module = ModelModule(self.loss)

        self._load_model(model_module, train_dataset, load_last_ckpt=load_last_ckpt)

        del train_dataset
        del val_dataset
        del train_dataloader
        del val_dataloader
        del test_datasets

        model_module.run_batch_inference(test_dataloaders, self.model_id, training_data_module.df, save_predictions, self.is_large)

    def run_evaluation(self):
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


        trading_algorithm = SimpleXDaysAheadBuying(interval_days=5, num_stocks_purchased=10, capital_gains_tax=0.35, uncertainty_multiplier=0.000, compound_money=False, dont_buy_negative_stocks=True, filter_out_x_most_volatile=200, predicting_raw_num=self.target in ['close', 'high', 'low', 'open'], pivot_df=filtered_df, day_of_week=DayOfWeek.tuesday)
        # trading_algorithm = VolatilityRanking(num_stocks_purchased=10, day_of_week=DayOfWeek.wednesday, volatility_importance=0.4)

        trading_algorithm.simulate(predictionsDF)


    ''' Setting testing to true does not gate any logic. All it does is move the day we run inference for back 
        till we reach a day the stock market was actually open. It thing skips any logic preventing trading earlier than intended.

        This is helpful for testing this workflow on days when the market isn't open like on Saturdays.'''
    def execute_buy(self, testing: bool=False):

        curr_day = get_prev_market_open_day()

        # get latest order details
        latest_order = self.db_util.get_latest_order(self.model_id)
        money_to_invest = 25000

        inference_data_module = InferenceDataModule(self.symbols, self.features + ['close', 'open', 'high', 'low', 'volume'], self.time_frame,
                                                 self.max_lookback_period, self.max_prediction_length, curr_date=curr_day)

        # determine trading strategy
        trading_strategy = self._init_trading_strategy()

        # sell previous order
        if latest_order is not None:
            order_items = [
                OrderItem(r.symbol, round(r.quantity, 2), OrderSide.SELL)
                for r in latest_order.order_items
            ]

            sell_order = Order(self.model_id, curr_day, order_items=order_items)
            self.db_util.upload_order(sell_order)
            self.order_util.close_all_positions()

        # execute trading strategy
        model_module = ModelModule(self.loss)

        with self.s3_util.load_best_model_checkpoint(self.model_id, pull_last_ckpt=True) as ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.accelerator, weights_only=False)
            hp = ckpt["hyper_parameters"]
            params = hp["dataset_parameters"]
        params["min_prediction_idx"] = None

        inf_dataset = inference_data_module.construct_inference_dataset(params)
        inf_dataloader = inference_data_module.construct_inference_dataloader(inf_dataset, self.batch_size, self.num_workers, self.use_gpu)

        model_module.load_model_from_checkpoint_and_data(self.model_id, self.accelerator,
                                                          inf_dataset, self.learning_rate,
                                                          self.hidden_size, self.attention_head_size,
                                                          self.dropout, self.hidden_continuous_size,
                                                          self.lstm_layers, self.reduce_on_plateau_patience, load_last_ckpt=True)

        predictionsDF = model_module.run_single_day_inference(inf_dataloader, inference_data_module.df)

        predictionsDF = model_module.append_actuals_to_simple_predictions(predictionsDF, inference_data_module.df)

        top_x = trading_strategy.generate_buy_list(predictionsDF, False)

        top_x['quantity'] = money_to_invest / top_x['close']

        order_items = [
            OrderItem(r.symbol, math.floor(r.quantity), OrderSide.BUY)
            for r in top_x.itertuples(index=False)
        ]

        # upload orders to db
        order = Order(self.model_id, datetime.now(), order_items=order_items)
        self.db_util.upload_order(order)

        # execute orders in alpaca
        self.order_util.place_order(order)

    def run_checkpoint_upload(self):
        logging.info("are you sure you meant to run checkpoint upload? Enter 'y' to continue: ")
        ans = input()
        if ans == "y":
            model_module = ModelModule(self.loss)

            model_module.upload_checkpoints_to_s3(self.model_id)
        else:
            logging.info('skipping checkpoint upload...')

    def _init_trading_strategy(self) -> BaseTradingModule:
        strat = self.config['preferred_trading_strategy']

        performance_test = self.config[strat]['_test_strategy']

        self.interval_days = self.config[strat]['_interval_days']

        filter_out_x_most_volatile = 0
        if '_filter_out_x_most_volatile' in self.config[strat]:
            filter_out_x_most_volatile = self.config[strat]['_filter_out_x_most_volatile']

        self.day_of_week = ''
        if '_day_of_week' in self.config[strat]:
            self.day_of_week = self.config[strat]['_day_of_week']


        if performance_test == 'SimpleXDaysAheadBuying':
            return SimpleXDaysAheadBuying(
                    interval_days=self.interval_days,
                    num_stocks_purchased=self.config[strat]['_num_stocks_purchased'],
                    capital_gains_tax=self.config[strat]['_capital_gains_tax'],
                    uncertainty_multiplier=self.config[strat]['_uncertainty_multiplier'],
                    compound_money=self.config[strat]['_compound_money'],
                    dont_buy_negative_stocks=self.config[strat]['_dont_buy_negative_stocks'],
                    filter_out_x_most_volatile=filter_out_x_most_volatile)
        else:
            raise Exception('The trading strategy specified is not supported')

    def _load_model(self, model_module: ModelModule, train_dataset: TimeSeriesDataSet, model_id='', modify_dropout=False, load_last_ckpt: bool=False):
        model_id = model_id if model_id != '' else self.model_id
        if not modify_dropout:
            try:
                model_module.load_model_from_checkpoint(model_id, self.accelerator, load_last_ckpt=load_last_ckpt)
            except:
                model_module.load_model_from_checkpoint_and_data(model_id, self.accelerator,
                                                                  train_dataset, self.learning_rate,
                                                                  self.hidden_size, self.attention_head_size,
                                                                  self.dropout, self.hidden_continuous_size,
                                                                  self.lstm_layers, self.reduce_on_plateau_patience,
                                                                  load_last_ckpt=load_last_ckpt)
        else:
            model_module.load_model_from_checkpoint_and_data(model_id, self.accelerator, 
                                                              train_dataset, self.learning_rate,
                                                              self.hidden_size, self.attention_head_size,
                                                              self.dropout, self.hidden_continuous_size,
                                                              self.lstm_layers, self.reduce_on_plateau_patience,
                                                              load_last_ckpt=load_last_ckpt)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbols_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/symbols.txt')
    parser.add_argument('--config_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/configs.yaml')
    parser.add_argument('--model_id', type=str, default='ubuntu-with-many-symbols-and-yfinance-and-more-complex')
    # 0 = False, 1 = True
    parser.add_argument('--run_training', type=bool, default=0)
    parser.add_argument('--run_batch_inference', type=bool, default=0)
    parser.add_argument('--run_evaluation', type=bool, default=1)

    parser.add_argument('--execute_buy', type=bool, default=0)

    # run_trainer uploads the checkpoints when complete. this function is useful for if we cancel training early we can still upload the models checkpoints to s3.
    parser.add_argument('--run_checkpoint_upload', type=bool, default=0)

    return parser.parse_args()


def main():

    logging.addLevelName(19, 'INFO_VERBOSE')

    logging.basicConfig(
        level=logging.INFO,
        # level='INFO_VERBOSE',
        format="%(levelname)s - %(message)s"
    )

    args = parse_args()

    with open('/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/many_symbols.txt', 'r') as f:
        symbols = [line.split('|')[0] for line in f]

    # symbols = symbols[:2500]

    # with open(args.symbols_path, "r") as f:
    #     symbols = [line.strip() for line in f]

    # when running locally
    if os.environ.get("SM_MODEL_DIR") is None:
        os.environ['SM_MODEL_DIR'] = './'

    orc = Orchestration(symbols, args.model_id, args.config_path)

    if args.run_training:
        orc.run_training()
    if args.run_batch_inference:
        orc.run_batch_inference(save_predictions=True, load_last_ckpt=True)
    if args.run_evaluation:
        orc.run_evaluation()
    if args.execute_buy:
        orc.execute_buy(True)
    if args.run_checkpoint_upload:
        orc.run_checkpoint_upload()




if __name__ == '__main__':
    main()

