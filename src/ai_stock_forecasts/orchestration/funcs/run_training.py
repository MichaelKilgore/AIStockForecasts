import os

import torch
from pytorch_forecasting import TemporalFusionTransformer

from ai_stock_forecasts.data.training_data_module import TrainingDataModule
from ai_stock_forecasts.model.lgbm_model_module import LgbmModelModule
from ai_stock_forecasts.model.tft_model_module import TftModelModule


def run_training(self):
    if self.model_type == 'lgbm':
        if self.resume_from_last_ckpt:
            raise NotImplementedError('resume_from_last_ckpt is only supported for tft models')
        _lgbm_run_training(self)
    elif self.model_type == 'tft':
        _tft_run_training(self)
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

    model_module = TftModelModule(self.loss)

    if self.fine_tuning_model_id:
        self._load_model(model_module, train_dataset, self.fine_tuning_model_id, modify_dropout=True, load_last_ckpt=True)
    elif self.resume_from_last_ckpt:
        ckpt_path = os.path.join(model_module.ckpt_dir, 'last.ckpt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'resume_from_last_ckpt set but no checkpoint found at {ckpt_path}')
        model_module.model = TemporalFusionTransformer.load_from_checkpoint(
            ckpt_path,
            map_location=torch.device(self.accelerator),
            weights_only=False,
        )

    model_module.run_training(train_dataset, self.learning_rate, self.hidden_size,
                               self.attention_head_size, self.dropout, self.hidden_continuous_size,
                               self.lstm_layers, self.reduce_on_plateau_patience, self.max_epochs,
                               self.accelerator, self.devices, train_dataloader,
                               val_dataloader, self.gradient_clip_val)

    model_module.upload_checkpoints_to_s3(self.model_id)

def _lgbm_run_training(self):
    training_data_module = TrainingDataModule(self.symbols, self.features,
                                               self.time_frame,
                                               -1,
                                               self.max_prediction_length,
                                               self.target,
                                               self.target_normalizer)

    model_module = LgbmModelModule()

    model_module.run_training(
        df=training_data_module.df,
        train_start=self.train_start,
        train_end=self.train_end,
        val_end=self.val_end,
        target=self.target,
        prediction_horizon=self.max_prediction_length,
        learning_rate=self.learning_rate,
        n_estimators=self.config.get('n_estimators', 500),
        num_leaves=self.config.get('num_leaves', 63),
        min_child_samples=self.config.get('min_child_samples', 20),
    )

    model_module.upload_model(self.model_id)

