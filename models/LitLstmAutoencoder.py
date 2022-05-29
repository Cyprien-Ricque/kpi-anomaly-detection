import logging
from logging import DEBUG

import pytorch_lightning as pl
import torch.nn.functional as F

logging.basicConfig(level=DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(DEBUG)

import torch
import torch.nn as nn

torch.manual_seed(0)


class Encoder(pl.LightningModule):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM1(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]
        return last_lstm_layer_hidden_state


class Decoder(pl.LightningModule):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class LSTM_AE(pl.LightningModule):
    def __init__(self, seq_len, no_features, embedding_dim):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features)

    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        raise NotImplementedError('predict_step is not implemented')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def _prepare_batch(self, batch):
        X, (y, _) = batch
        seq_true = X['encoder_cont']
        filter_ = X['encoder_cat'].to(torch.bool)[:, :, :1]
        return filter_, seq_true

    def _common_step(self, batch, batch_idx, stage: str):
        filter_, seq_true = self._prepare_batch(batch)
        y_hat = self(seq_true)
        loss = F.l1_loss(y_hat[filter_], seq_true[filter_], reduction='sum')
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


if __name__ == '__main__':
    model = LSTM_AE(seq_len=118, no_features=4, embedding_dim=64)

    data = torch.rand(5, 118, 4)
    print(data.shape)
    a = model(data)
    print(a.shape)
