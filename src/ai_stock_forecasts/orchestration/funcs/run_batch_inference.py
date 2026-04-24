
from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.model.model_module import ModelModule


def run_batch_inference(self, *args):
    if self.model_type == 'lgbm':
        self._lgbm_run_batch_inference(self)
    elif self.model_type == 'tft':
        self._tft_run_batch_inference(self, args)
    else:
        raise Exception(f'model type: {self.model_type} not supported')

def _tft_run_batch_inference(self, save_predictions=True, load_last_ckpt=False):
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


def _lgbm_run_batch_inference(self):
    pass


