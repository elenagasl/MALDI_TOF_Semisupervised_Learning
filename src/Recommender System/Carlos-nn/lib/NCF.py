import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchmetrics import AUROC


# =========================
# MALDI EMBEDDING (MLP FIJA)
# =========================
class Em_Spectrum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.net(x.float())


# =========================
# GENERIC EMBEDDING
# =========================
class Embedding(nn.Module):
    def __init__(self, num_embeddings, dim_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, dim_embeddings)

    def forward(self, x):
        return self.embedding(x.long())


# =========================
# NCF MODEL
# =========================
class NCF(pl.LightningModule):

    def __init__(
        self,
        num_feat,
        num_items,
        num_hospitals,
        num_sample_types,
        hidden_dim_CF=[64, 32]
    ):
        super().__init__()

        # =========================
        # EMBEDDINGS
        # =========================
        self.embedding_s = Em_Spectrum(num_feat)         # → 64
        self.embedding_d = Embedding(num_items, 30)      # → 30
        self.embedding_h = Embedding(num_hospitals, 10)  # → 10
        self.embedding_st = Embedding(num_sample_types, 10)  # → 10

        # =========================
        # CF NETWORK
        # =========================
        input_dim = 64 + 30 + 10 + 10  # = 114

        sizes = [input_dim] + list(hidden_dim_CF) + [1]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())

        self.CF_net = nn.Sequential(*layers)

        # =========================
        # METRICS
        # =========================
        self.acc_tr = Accuracy(task="binary")
        self.acc_val = Accuracy(task="binary")
        self.acc_tst = Accuracy(task="binary")

        self.auc_tr = AUROC(task='binary')
        self.auc_val = AUROC(task='binary')
        self.auc_tst = AUROC(task='binary')

        self.loss_fn = nn.BCELoss()

    # =========================
    # FORWARD
    # =========================
    def forward(self, maldi, hospital, sample_type, item):

        maldi_emb = self.embedding_s(maldi)
        item_emb = self.embedding_d(item)
        hospital_emb = self.embedding_h(hospital)
        stype_emb = self.embedding_st(sample_type)

        x = torch.cat([maldi_emb, item_emb, hospital_emb, stype_emb], dim=-1)

        return self.CF_net(x)

    # =========================
    # TRAINING
    # =========================
    def training_step(self, batch, batch_idx):

        maldi, hospital, sample_type, item, labels = batch

        preds = self.forward(maldi, hospital, sample_type, item)

        loss = self.loss_fn(preds, labels.view(-1, 1))

        self.acc_tr(preds, labels.view(-1, 1).int())
        self.auc_tr.update(preds, labels.view(-1, 1))

        self.log("loss_tr", loss)

        return loss

    def on_train_epoch_end(self):
        self.log_dict({
            "acc_tr": self.acc_tr,
            "auc_tr": self.auc_tr.compute()
        }, prog_bar=True)

        self.auc_tr.reset()

    # =========================
    # VALIDATION
    # =========================
    def validation_step(self, batch, batch_idx):

        maldi, hospital, sample_type, item, labels = batch

        preds = self.forward(maldi, hospital, sample_type, item)

        loss = self.loss_fn(preds, labels.view(-1, 1))

        self.acc_val(preds, labels.view(-1, 1).int())
        self.auc_val.update(preds, labels.view(-1, 1))

        self.log("loss_val", loss)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict({
            "acc_val": self.acc_val,
            "auc_val": self.auc_val.compute()
        }, prog_bar=True)

        self.auc_val.reset()

    # =========================
    # TEST
    # =========================
    def test_step(self, batch, batch_idx):

        maldi, hospital, sample_type, item, labels = batch

        preds = self.forward(maldi, hospital, sample_type, item)

        loss = self.loss_fn(preds, labels.view(-1, 1))

        self.acc_tst(preds, labels.view(-1, 1).int())
        self.auc_tst.update(preds, labels.view(-1, 1))

        self.log("loss_tst", loss)

        return loss

    def on_test_epoch_end(self):
        self.log_dict({
            "acc_tst": self.acc_tst,
            "auc_tst": self.auc_tst.compute()
        }, prog_bar=True)

        self.auc_tst.reset()

    # =========================
    # OPTIMIZER
    # =========================
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)