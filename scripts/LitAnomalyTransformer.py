
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_factory.DataLoader import DataLoader
from utils.config import load_config
# evaluation file
from utils.evaluation import label_evaluation

import numpy as np
from tqdm import tqdm

# sys.path.append('/media/cyprien/Data/Documents/Github/pytorch-forecasting')
sys.path.append('../../pytorch-forecasting')


config_file = "../config/config.yml"
config = load_config(config_file)

result_file = '../predict.csv'
truth_file = '../ground_truth.hdf'


dl = DataLoader(use_previous_files=True, config_file=config_file)


max_prediction_length = 1
max_encoder_length = config['AnomalyTransformer']['max_seq_len']
min_encoder_length = config['AnomalyTransformer']['min_seq_len']


from pytorch_forecasting import TimeSeriesDataSet

X_cols = ['value_scaled', 'kpi_id', 'timestamp_1', 'authentic']
normal_train = dl.train.reset_index(drop=True)

training = TimeSeriesDataSet(
    normal_train.loc[:, X_cols],
    time_idx='timestamp_1', target='value_scaled',
    group_ids=['kpi_id'],
    allow_missing_timesteps=False,
    static_categoricals=['kpi_id', 'authentic'],
    time_varying_unknown_reals=['value_scaled'],
    # time_varying_known_reals=['timestamp_1'],
    max_encoder_length=max_encoder_length,
    min_encoder_length=min_encoder_length,
    max_prediction_length=max_prediction_length,
    scalers={col: None for col in ['timestamp_1', 'kpi_id']},
    target_normalizer=None,
    add_relative_time_idx=False,
    add_target_scales=False,
    add_encoder_length=False,
)

validation = TimeSeriesDataSet.from_dataset(
    training, dl.val.loc[:, X_cols], stop_randomization=True, predict=False
)
testing = TimeSeriesDataSet.from_dataset(
    training, dl.test.loc[:, X_cols], stop_randomization=True, predict=False, min_encoder_length=max_encoder_length
)


batch_size = 64

training_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=12)

validation_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=12)
testing_dl = testing.to_dataloader(train=False, batch_size=batch_size * 3, num_workers=12)



from models.AnomalyTransformer.AnomalyTransformer import AnomalyTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
n_features = 1


from tqdm import tqdm

model = AnomalyTransformer(
    win_size=max_encoder_length, enc_in=n_features, c_out=1,
    d_model=256, n_heads=4, e_layers=2, d_ff=256,
    dropout=0.0, activation='gelu', output_attention=True
)

# model = LitAE.load_from_checkpoint("./lightning_logs/version_1/checkpoints/epoch=4-step=171145.ckpt",
#                                    input_shape=max_encoder_length, n_dim=n_dim)


# In[11]:


from pytorch_lightning.trainer import Trainer

trainer = Trainer(logger=True, enable_checkpointing=True, checkpoint_callback=None, gpus=1, auto_lr_find=True, max_epochs=-1)

trainer.validate(model=model, dataloaders=validation_dl)


# In[12]:


trainer.fit(model=model, train_dataloaders=training_dl, val_dataloaders=validation_dl)


# In[13]:


trainer.validate(model=model, dataloaders=validation_dl)


def predict_value(df: pd.DataFrame):
    df['timestamp_1_floor'] = df.groupby('kpi_id').timestamp_1.transform(lambda x: x - x.min())
    pv = pd.pivot_table(df, values='value_scaled', index='timestamp_1_floor', columns='kpi_id', fill_value=np.nan)

    ept = np.empty(max_encoder_length)
    ept[:] = np.nan

    pv = pd.concat([pv, pd.DataFrame({col: ept.copy() for col in pv.columns})])

    pv_forward = np.moveaxis(pv.copy().to_numpy(dtype=np.float32), 1, 0)
    pv_filter = np.moveaxis(pv.copy().to_numpy(), 1, 0)
    pv_forward[np.isnan(pv_forward)] = 0
    pv_filter = np.where(np.isnan(pv_filter), 0, 1)

    df['value_pred'] = np.nan

    for i in tqdm(np.arange(0, pv.index.max() + 1, max_encoder_length)):
        pv_forward_i = pv_forward[:, i:i + max_encoder_length]
        pv_filter_i = pv_filter[:, i:i + max_encoder_length]

        X = torch.from_numpy(pv_forward_i.copy())
        X = X.reshape(X.shape[0], X.shape[1], 1)

        y_pred, _, _, _ = model(X.to(device))

        validation_filter = df.groupby('kpi_id').apply(
            lambda x: (x.timestamp_1_floor >= i) & (x.timestamp_1_floor < i + max_encoder_length)).reset_index(
            drop=True)

        df.loc[validation_filter, 'value_pred'] = y_pred.cpu().detach().numpy()[pv_filter_i.astype(bool)].flatten()


predict_value(dl.train)
predict_value(dl.test)

import pickle

with open('../dataloader.p', 'wb') as f:
    pickle.dump(dl, f)
