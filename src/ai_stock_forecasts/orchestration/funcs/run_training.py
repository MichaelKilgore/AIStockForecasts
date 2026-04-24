
from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.model.model_module import ModelModule


def run_training(self):
    if self.model_type == 'lgbm':
        self._lgbm_run_training(self)
    elif self.model_type == 'tft':
        self._tft_run_training(self)
    else:
        raise Exception(f'model type: {self.model_type} not supported')

def _tft_run_training(self):
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

def _lgbm_run_training(self):
    pass

