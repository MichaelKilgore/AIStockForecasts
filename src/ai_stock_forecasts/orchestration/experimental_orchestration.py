
import argparse
import os
from ai_stock_forecasts.orchestration.orchestration import Orchestration
from ai_stock_forecasts.data.inference_data_module import InferenceDataModule
from ai_stock_forecasts.model.model_module import ModelModule
from ai_stock_forecasts.trading_algorithms.simple_x_days_ahead_buying import SimpleXDaysAheadBuying

from ai_stock_forecasts.data.training_data_module import TrainingDataModule


"""
    Extends off of main Orchestration class. This class mainly just holds some less commonly used activities
    that we use for testing the model. But we use them less often than the core functions:
        - run_training, run_batch_inference, run_exaluation, execute_buy, and run_checkpoint_upload

    for simplicity all the code bloat is moved to this class.
"""
class ExperimentalOrchestration(Orchestration):
    def __init__(self, symbols: list[str], model_id: str, config_path: str):
        super().__init__(symbols, model_id, config_path)

    def explain_model(self):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                   self.time_frame,
                                                   self.max_lookback_period, self.max_prediction_length,
                                                   self.target, self.target_normalizer)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, _ = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        test_datasets = training_data_module.construct_test_datasets(train_dataset, self.train_start, self.val_end, self.test_end)
        tst_dataloaders = training_data_module.construct_test_dataloaders(test_datasets, self.batch_size, self.num_workers, self.use_gpu)

        model_module = ModelModule(self.loss)

        self._load_model(model_module, train_dataset)

        model_module.interpret_predictions(train_dataloader)

    def run_inference(self):
        inference_data_module = InferenceDataModule(self.symbols, self.features, self.time_frame, 
                                                     self.max_lookback_period, self.max_prediction_length)

        model_module = ModelModule(self.loss)

        # TODO: Sometimes when models are trained on different hardware, loading in the model this way doesn't work.
        model_module.load_model_from_checkpoint(self.model_id, self.accelerator)

        inf_dataset = inference_data_module.construct_inference_dataset(model_module.model.hparams["dataset_parameters"])
        inf_dataloader = inference_data_module.construct_inference_dataloader(inf_dataset, self.batch_size, self.num_workers, self.use_gpu)

        predictionsDF = model_module.run_single_day_inference(inf_dataloader, inference_data_module.df)

        trading_algorithm = SimpleXDaysAheadBuying(interval_days=2, num_stocks_purchased=50, capital_gains_tax=0.35, uncertainty_multiplier=0.3, dont_buy_negative_stocks=True)

        stocks = trading_algorithm.generate_buy_list(predictionsDF)

        print(stocks)

    def find_optimal_hyperparams(self):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                   self.time_frame,
                                                   self.max_lookback_period, self.max_prediction_length)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, val_dataloader = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        if self.config['devices'] > 1:
            training_data_module.cache_df()

        model_module = ModelModule(self.loss)

        model_module.find_optimal_hyperparameters(train_dataloader, val_dataloader)

    def find_optimal_learning_rate(self):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                   self.time_frame,
                                                   self.max_lookback_period, self.max_prediction_length)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, val_dataloader = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        if self.config['devices'] > 1:
            training_data_module.cache_df()

        model_module = ModelModule(self.loss)

        self._load_model(model_module, train_dataset)

        model_module.find_optimal_learning_rate(train_dataloader,
                                                 val_dataloader,
                                                 self.max_epochs, self.accelerator, self.devices)

    def plot_predictions(self):
        training_data_module = TrainingDataModule(self.symbols, self.features,
                                                   self.time_frame,
                                                   self.max_lookback_period,
                                                   self.max_prediction_length,
                                                   self.target,
                                                   self.target_normalizer)

        train_dataset, val_dataset = training_data_module.construct_training_and_validation_datasets(self.train_start, self.train_end, self.val_end)
        train_dataloader, val_dataloader = training_data_module.construct_train_and_validation_dataloaders(train_dataset, val_dataset, self.batch_size, self.num_workers, self.use_gpu)

        test_datasets = training_data_module.construct_test_datasets(train_dataset, self.train_start, self.val_end, self.test_end)
        test_dataloaders = training_data_module.construct_test_dataloaders(test_datasets, self.batch_size, self.num_workers, self.use_gpu)

        model_module = ModelModule(self.loss)

        self._load_model(model_module, train_dataset)

        model_module.plot_prediction(test_dataloaders[0])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbols_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/symbols.txt')
    parser.add_argument('--config_path', type=str, default='/home/michael/Coding/AIStockForecasts/src/ai_stock_forecasts/constants/configs.yaml')
    parser.add_argument('--model_id', type=str, default='ubuntu-with-many-symbols-and-yfinance')
    # 0 = False, 1 = True
    parser.add_argument('--run_training', type=bool, default=0)
    parser.add_argument('--run_batch_inference', type=bool, default=0)
    parser.add_argument('--run_evaluation', type=bool, default=0)
    parser.add_argument('--explain_model', type=bool, default=0)

    parser.add_argument('--run_inference', type=bool, default=0)
    parser.add_argument('--execute_buy', type=bool, default=1)
    parser.add_argument('--find_optimal_hyperparams', type=bool, default=0)
    parser.add_argument('--find_optimal_learning_rate', type=bool, default=0)
    parser.add_argument('--plot_prediction', type=bool, default=0)

    # run_trainer uploads the checkpoints early but its helpful for when you stop training early you can run checkpoint upload early.
    parser.add_argument('--run_checkpoint_upload', type=bool, default=0)

    return parser.parse_args()


def main():
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
    if args.explain_model:
        orc.explain_model()
    if args.run_inference:
        orc.run_inference()
    if args.execute_buy:
        orc.execute_buy(True)
    if args.find_optimal_hyperparams:
        orc.find_optimal_hyperparams()
    if args.find_optimal_learning_rate:
        orc.find_optimal_learning_rate()
    if args.plot_prediction:
        orc.plot_predictions()
    if args.run_checkpoint_upload:
        orc.run_checkpoint_upload()




if __name__ == '__main__':
    main()
