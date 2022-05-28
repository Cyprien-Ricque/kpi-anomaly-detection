import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding


class EncoderLayer(pl.LightningModule):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(pl.LightningModule):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(pl.LightningModule):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, k=3):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.win_size = win_size
        self.k = k

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(self.win_size, False, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        min_loss, max_loss = self._common_step(batch, batch_idx, "train")

        min_loss.backward(retain_graph=True)
        # max_loss.backward()
        # self.optimizer.step()

        return {'loss': max_loss, 'min_loss': min_loss, 'max_loss': max_loss}

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        raise NotImplementedError('predict_step is not implemented')

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        return [self.optimizer], [self.lr_scheduler]

    def _prepare_batch(self, batch):
        X, (y, _) = batch
        seq_true = X['encoder_cont']
        filter_ = X['encoder_cat'].to(torch.bool)[:, :, :1]
        return filter_, seq_true

    def _common_step(self, batch, batch_idx, stage: str):
        filter_, seq_true = self._prepare_batch(batch)
        output, series, prior, _ = self(seq_true)

        min_loss, max_loss = self.min_max_loss(seq_true, output, prior, series)

        self.log(f"{stage}_loss", min_loss + max_loss, on_step=True, batch_size=seq_true.shape[0])
        self.log(f"{stage}_min_loss", min_loss, on_step=True, batch_size=seq_true.shape[0])
        self.log(f"{stage}_max_loss", max_loss, on_step=True, batch_size=seq_true.shape[0])
        return min_loss, max_loss

    def my_kl_loss(self, p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def min_max_loss(self, input, output, prior, series):
        # calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(self.my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                self.my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)).detach(),
                                series[u])))
            prior_loss += (torch.mean(self.my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)),
                series[u].detach())) + torch.mean(self.my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        # print(output.shape, input.shape)
        rec_loss = F.mse_loss(output, input)
        loss1 = rec_loss - self.k * series_loss
        loss2 = rec_loss + self.k * prior_loss
        return loss1, loss2


if __name__ == '__main__':
    model = AnomalyTransformer(
        win_size=100, enc_in=3, c_out=1,
        d_model=512, n_heads=8, e_layers=3, d_ff=512,
        dropout=0.0, activation='gelu', output_attention=False
    ).cpu()

    x = torch.ones(10, 100, 3).to(torch.float).cpu()
    print(x.device)
    out = model(x)

