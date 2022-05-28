import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class LitAE(pl.LightningModule):
    def __init__(self, input_shape, n_dim=128):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=n_dim)
        self.encoder_output_layer = nn.Linear(in_features=n_dim, out_features=n_dim)
        self.decoder_hidden_layer = nn.Linear(in_features=n_dim, out_features=n_dim)
        self.decoder_output_layer = nn.Linear(in_features=n_dim, out_features=input_shape)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
        # reconstructed = torch.tanh(activation)
        return reconstructed

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        raise NotImplementedError('predict_step is not implemented')

    def _prepare_batch(self, batch):
        X, (y, _) = batch
        seq_true = X['encoder_cont'].squeeze(2)
        filter_ = X['encoder_cat'].squeeze(2).to(torch.bool)[:, :, 1]
        return filter_, seq_true

    def _common_step(self, batch, batch_idx, stage: str):
        filter_, seq_true = self._prepare_batch(batch)
        y_hat = self(seq_true)
        loss = F.l1_loss(y_hat[filter_], seq_true[filter_], reduction='sum')
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

