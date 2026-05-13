
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
from ai_stock_forecasts.model.tft_model_module import TftModelModule
from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule
from ai_stock_forecasts.trading_algorithms.simple_x_days_ahead_buying import SimpleXDaysAheadBuying
import sys

from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.models.order import Order, OrderItem

from alpaca.trading.enums import OrderSide

from ai_stock_forecasts.ordering.order_util import OrderUtil
from ai_stock_forecasts.utils.postgres_util import PostgresUtil
from ai_stock_forecasts.utils.s3_util import S3ParquetUtil
from ai_stock_forecasts.utils.yfinance_util import YfinanceUtil

import pandas as pd
import numpy as np
import math

import logging

from ai_stock_forecasts.orchestration.funcs.run_training import run_training
from ai_stock_forecasts.orchestration.funcs.run_batch_inference import run_batch_inference
from ai_stock_forecasts.orchestration.funcs.run_evaluation import run_evaluation



class Orchestration:
    def __init__(self, symbols: list[str], model_id: str, config_path: str):
        self.model_id = model_id

        self._setConfig(config_path)

        self.test_model: bool = self.config.get('test_model', False)
        self.symbols = ['AAPL'] if self.test_model else symbols

        self.train_start = datetime.fromisoformat(self.config['train_start']).replace(tzinfo=timezone.utc)
        self.train_end = datetime.fromisoformat(self.config['train_end']).replace(tzinfo=timezone.utc)
        self.val_end = (
            datetime.fromisoformat(self.config['val_end']).replace(tzinfo=timezone.utc)
            if self.config.get('val_end') else None
        )
        self.test_end = (
            datetime.fromisoformat(self.config['test_end']).replace(tzinfo=timezone.utc)
            if self.config.get('test_end') else None
        )

        self.features: list[str] = self.config['features']

        if sys.platform == 'darwin':
            self.accelerator: str = 'mps'
        else:
            self.accelerator: str = self.config['accelerator']

        time_frame_amount: int = self.config['time_frame_amount']
        time_frame_unit: str = self.config['time_frame_unit']
        self.time_frame: TimeFrame = TimeFrame(time_frame_amount, TimeFrameUnit(time_frame_unit))

        self.max_prediction_length: int = self.config['max_prediction_length']

        self.use_gpu = self.accelerator in ('gpu', 'cuda', 'mps')

        self.learning_rate: float = self.config['learning_rate']

        self.db_util = DynamoDBUtil()
        self.order_util = OrderUtil()
        self.s3_util = S3ParquetUtil()
        self.postgres_util = PostgresUtil()
        self.yfinance_util = YfinanceUtil()

        if 'target' in self.config:
            self.target = self.config['target']
        else:
            self.target = 'open'

        if 'target_normalizer' in self.config:
            self.target_normalizer = self.config['target_normalizer']
        else:
            self.target_normalizer = 'auto'

        self.model_type = self.config.get('model_type', 'tft')

        if self.model_type == 'tft':
            self._init_tft_args()

    def _init_tft_args(self):
        self.max_lookback_period: int = self.config['max_lookback_period']

        self.devices: int = self.config['devices']
        self.batch_size: int = self.config['batch_size']
        self.num_workers: int = self.config['num_workers']

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

        self.quantiles: list[float] = self.config['quantiles'] if 'quantiles' in self.config else [0.3, 0.5, 0.7]

        if 'loss' not in self.config:
            self.loss = QuantileLoss(quantiles=self.quantiles)
        elif self.config['loss'] == 'WeightedQuantileLoss':
            self.loss = WeightedQuantileLoss(quantiles=self.quantiles)

        if 'gradient_clip_val' not in self.config:
            self.gradient_clip_val = None
        else:
            self.gradient_clip_val = self.config['gradient_clip_val']

        self.is_large = False
        if 'is_large' in self.config:
            self.is_large = self.config['is_large']

    def _setConfig(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f) or {}

        if 'configs' not in full_config:
            raise Exception(f'the config path: {config_path} passed is not valid')

        if self.model_id not in full_config['configs']:
            raise Exception(f'The config id: {self.model_id} passed is not defined in configurations')

        raw_config = full_config['configs'][self.model_id]

        if 'variant_of' in raw_config:
            base_id = raw_config['variant_of']
            strat_key = raw_config.get('preferred_trading_strategy')

            if strat_key is None:
                raise Exception(f'Variant model {self.model_id} must set preferred_trading_strategy')

            allowed_keys = {'variant_of', 'preferred_trading_strategy', strat_key}
            extra = set(raw_config.keys()) - allowed_keys
            if extra:
                raise Exception(
                    f'Variant model {self.model_id} may only override preferred_trading_strategy; '
                    f'got extra keys: {sorted(extra)}'
                )

            if base_id not in full_config['configs']:
                raise Exception(f'Variant {self.model_id} references unknown base model {base_id}')

            base_config = full_config['configs'][base_id]
            if 'variant_of' in base_config:
                raise Exception(
                    f'Variant {self.model_id} points to {base_id}, which is itself a variant. '
                    f'Variant-of-variant is not allowed.'
                )

            self.is_variant = True
            self.base_model_id = base_id
            self.config = {**base_config, 'preferred_trading_strategy': strat_key, strat_key: raw_config[strat_key]}
        else:
            self.is_variant = False
            self.base_model_id = self.model_id
            self.config = raw_config

    def _assert_not_variant(self, stage: str):
        if self.is_variant:
            raise Exception(
                f'{stage} is not supported for variant model {self.model_id} '
                f'(variant_of={self.base_model_id}). Run this stage against the base model.'
            )

    def run_training(self, resume_from_last_ckpt: bool = False):
        self._assert_not_variant('run_training')
        self.resume_from_last_ckpt = resume_from_last_ckpt
        run_training(self)

    def run_batch_inference(self, save_predictions=True, load_last_ckpt=False):
        self._assert_not_variant('run_batch_inference')
        run_batch_inference(self, save_predictions, load_last_ckpt)

    def run_evaluation(self):
        self._assert_not_variant('run_evaluation')
        run_evaluation(self)

    ''' Setting testing to true skips db upload and actually execute sell / buy in alpaca.'''
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
        if latest_order is not None and not testing:
            order_items = [
                OrderItem(r.symbol, round(r.quantity, 2), OrderSide.SELL)
                for r in latest_order.order_items
            ]

            sell_order = Order(self.model_id, curr_day, order_items=order_items)
            self.db_util.upload_order(sell_order)
            self.order_util.close_all_positions()

            sell_timestamp = datetime.now()
            sell_prices = {
                stock.symbol: stock.price
                for stock in self.yfinance_util.get_current_prices([oi.symbol for oi in order_items])
            }
            for oi in order_items:
                self.postgres_util.add_transaction(
                    model_id=self.model_id,
                    symbol=oi.symbol,
                    timestamp=sell_timestamp,
                    price=sell_prices[oi.symbol],
                    count=int(oi.quantity),
                    side='sell',
                )

        # execute trading strategy
        model_module = TftModelModule(self.loss)

        with self.s3_util.load_best_model_checkpoint(self.base_model_id, pull_last_ckpt=self.pull_last_ckpt) as ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.accelerator, weights_only=False)
            hp = ckpt["hyper_parameters"]
            params = hp["dataset_parameters"]
        params["min_prediction_idx"] = None

        inf_dataset = inference_data_module.construct_inference_dataset(params)
        inf_dataloader = inference_data_module.construct_inference_dataloader(inf_dataset, self.batch_size, self.num_workers, self.use_gpu)

        model_module.load_model_from_checkpoint_and_data(self.base_model_id, self.accelerator,
                                                          inf_dataset, self.learning_rate,
                                                          self.hidden_size, self.attention_head_size,
                                                          self.dropout, self.hidden_continuous_size,
                                                          self.lstm_layers, self.reduce_on_plateau_patience, load_last_ckpt=self.pull_last_ckpt)

        predictionsDF = model_module.run_single_day_inference(inf_dataloader, inference_data_module.df)

        predictionsDF = model_module.append_actuals_to_simple_predictions(predictionsDF, inference_data_module.df)

        top_x = trading_strategy.generate_buy_list(predictionsDF)

        money_per_symbol = money_to_invest / len(top_x)
        top_x['quantity'] = money_per_symbol / top_x['close']

        holding_period_days = 5
        top_x['expected_future_price'] = top_x.apply(
            lambda r: r['close'] * np.exp(np.sum(r['y_pred_p50'][:min(holding_period_days, len(r['y_pred_p50']))])),
            axis=1
        )
        top_x['limit_price'] = ((top_x['close'] + top_x['expected_future_price']) / 2).round(2)

        print('TESTING')
        print(top_x)

        order_items = [
            OrderItem(r.symbol, math.floor(r.quantity), OrderSide.BUY, limit_price=round(float(r.limit_price), 2))
            for r in top_x.itertuples(index=False)
        ]

        buy_prices = { r.symbol: float(r.close) for r in top_x.itertuples(index=False) }

        if not testing:
            # upload orders to db
            buy_timestamp = datetime.now()
            order = Order(self.model_id, buy_timestamp, order_items=order_items)
            self.db_util.upload_order(order)

            # execute orders in alpaca
            self.order_util.place_order(order)

            for oi in order_items:
                self.postgres_util.add_transaction(
                    model_id=self.model_id,
                    symbol=oi.symbol,
                    timestamp=buy_timestamp,
                    price=buy_prices[oi.symbol],
                    count=int(oi.quantity),
                    side='buy',
                )

    def run_checkpoint_upload(self):
        self._assert_not_variant('run_checkpoint_upload')
        logging.info("are you sure you meant to run checkpoint upload? Enter 'y' to continue: ")
        ans = input()
        if ans == "y":
            model_module = TftModelModule(self.loss)

            model_module.upload_checkpoints_to_s3(self.model_id)
        else:
            logging.info('skipping checkpoint upload...')

    def _init_trading_strategy(self) -> BaseTradingModule:
        strat = self.config['preferred_trading_strategy']

        performance_test = self.config[strat]['_test_strategy']

        self.interval_days = self.config[strat].get('_interval_days')

        filter_out_x_most_volatile = 0
        if '_filter_out_x_most_volatile' in self.config[strat]:
            filter_out_x_most_volatile = self.config[strat]['_filter_out_x_most_volatile']

        self.day_of_week = ''
        if '_day_of_week' in self.config[strat]:
            self.day_of_week = self.config[strat]['_day_of_week']

        self.pull_last_ckpt = self.config[strat].get('_pull_last_ckpt', False)

        if performance_test == 'SimpleXDaysAheadBuying':
            return SimpleXDaysAheadBuying(
                    interval_days=self.interval_days,
                    num_stocks_purchased=self.config[strat]['_num_stocks_purchased'],
                    uncertainty_multiplier=self.config[strat]['_uncertainty_multiplier'],
                    filter_out_x_most_volatile=filter_out_x_most_volatile)
        elif performance_test == 'VolatilityRanking':
            day_of_week = DayOfWeek(self.day_of_week) if self.day_of_week else DayOfWeek.tuesday
            volatility_importance = self.config[strat].get('_volatility_importance', 0.3)
            return VolatilityRanking(
                    num_stocks_purchased=self.config[strat]['_num_stocks_purchased'],
                    day_of_week=day_of_week,
                    volatility_importance=volatility_importance)
        else:
            raise Exception('The trading strategy specified is not supported')

    def _load_model(self, model_module: TftModelModule, train_dataset: TimeSeriesDataSet, model_id='', modify_dropout=False, load_last_ckpt: bool=False):
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


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1', 'yes', 'y'):
        return True
    if v.lower() in ('false', 'f', '0', 'no', 'n', ''):
        return False
    raise argparse.ArgumentTypeError(f'expected boolean, got {v!r}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbols_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/symbols.txt')
    parser.add_argument('--config_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/configs.yaml')
    parser.add_argument('--model_id', type=str, default='ubuntu-with-long-training')
    parser.add_argument('--run_training', type=_str2bool, default=True)
    parser.add_argument('--resume_from_last_ckpt', type=_str2bool, default=False)
    parser.add_argument('--run_batch_inference', type=_str2bool, default=False)
    parser.add_argument('--run_evaluation', type=_str2bool, default=False)

    parser.add_argument('--execute_buy', type=_str2bool, default=False)
    parser.add_argument('--testing', type=_str2bool, default=False)

    # run_trainer uploads the checkpoints when complete. this function is useful for if we cancel training early we can still upload the models checkpoints to s3.
    parser.add_argument('--run_checkpoint_upload', type=_str2bool, default=False)

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

    # when running locally
    if os.environ.get("SM_MODEL_DIR") is None:
        os.environ['SM_MODEL_DIR'] = './'

    orc = Orchestration(symbols, args.model_id, args.config_path)

    if args.run_training:
        orc.run_training(args.resume_from_last_ckpt)
    if args.run_batch_inference:
        orc.run_batch_inference(save_predictions=True, load_last_ckpt=True)
    if args.run_evaluation:
        orc.run_evaluation()
    if args.execute_buy:
        orc.execute_buy(args.testing)
    if args.run_checkpoint_upload:
        orc.run_checkpoint_upload()




if __name__ == '__main__':
    main()

