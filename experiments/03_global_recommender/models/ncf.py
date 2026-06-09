"""
Neural Collaborative Filtering (NCF) for MALDI-TOF AMR Prediction
==================================================================

Implements the species-specific NCF model that combines:
    - A MALDI-TOF spectrum encoder (MLP: 6000 → 512 → 256 → 128 → 64)
    - A target antibiotic embedding (learnable, size drug_emb_dim)
    - An antibiotic family embedding (learnable, size family_emb_dim)
    - An AMR context encoder that summarises the known resistance profile
      of the sample across all antibiotics, passing both values and a
      binary observation mask to distinguish true zeros from missing entries

All four representations are concatenated and passed through the fusion
network (CF_net) to produce P(resistant | spectrum, antibiotic, context).

This corresponds to the species-specific recommender baseline described
in Paper Objective 2.

Classes
-------
Em_Spectrum : nn.Module
    MALDI-TOF spectrum encoder backbone shared across antibiotic predictions.

NCF : pl.LightningModule
    Full Neural Collaborative Filtering model with AMR context.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


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


class NCF(pl.LightningModule):

    def __init__(
        self,
        num_feat,
        num_items,
        num_families,
        drug_emb_dim=32,
        family_emb_dim=8,
        hidden_dim_CF=[128, 64],
        lr=1e-3,
        amr_dropout=0.1
    ):
        super().__init__()

        self.num_items = num_items
        self.num_families = num_families
        self.lr = lr

        # =========================
        # MALDI encoder
        # =========================
        self.embedding_s = Em_Spectrum(num_feat)

        # =========================
        # Target antibiotic embedding
        # =========================
        self.embedding_d = nn.Embedding(num_items, drug_emb_dim)

        # =========================
        # Target family embedding
        # =========================
        self.embedding_f = nn.Embedding(num_families, family_emb_dim)

        # =========================
        # AMR context encoder (values + mask)
        # =========================
        self.amr_encoder = nn.Sequential(
            nn.Linear(num_items * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU()
        )

        # Lower dropout on AMR context to preserve contextual signal
        self.amr_dropout = nn.Dropout(amr_dropout)

        # =========================
        # Fusion network
        # =========================
        input_dim = 64 + drug_emb_dim + family_emb_dim + 64
        # MALDI emb + drug emb + family emb + AMR context emb

        sizes = [input_dim] + list(hidden_dim_CF) + [1]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())

        self.CF_net = nn.Sequential(*layers)

        # =========================
        # Binary loss
        # =========================
        self.loss_fn = nn.BCELoss()

    # =========================
    # FORWARD
    # =========================
    def forward(self, maldi, target_drug, target_family, amr_vec, mask):

        maldi_emb = self.embedding_s(maldi)

        drug_emb = self.embedding_d(target_drug.long())
        family_emb = self.embedding_f(target_family.long())

        # Apply mask so only observed antibiotic labels enter as context
        amr_vec = amr_vec * mask

        # Concatenate values and mask so the encoder can distinguish true 0 (susceptible) from missing
        amr_input = torch.cat([amr_vec, mask], dim=-1)

        amr_emb = self.amr_encoder(amr_input)
        amr_emb = self.amr_dropout(amr_emb)

        x = torch.cat([
            maldi_emb,
            drug_emb,
            family_emb,
            amr_emb
        ], dim=-1)

        return self.CF_net(x)

    # =========================
    # GPU helper
    # =========================
    def move_batch(self, batch):
        device = self.device
        return [x.to(device) for x in batch]

    # =========================
    # TRAINING
    # =========================
    def training_step(self, batch, batch_idx):

        maldi, target_drug, target_family, labels, amr_vec, mask = self.move_batch(batch)

        preds = self.forward(
            maldi,
            target_drug,
            target_family,
            amr_vec,
            mask
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(preds, labels)

        self.log("loss_tr", loss, prog_bar=True)
        return loss

    # =========================
    # VALIDATION
    # =========================
    def validation_step(self, batch, batch_idx):

        maldi, target_drug, target_family, labels, amr_vec, mask = self.move_batch(batch)

        preds = self.forward(
            maldi,
            target_drug,
            target_family,
            amr_vec,
            mask
        )

        labels = labels.view(-1, 1).float()

        loss = self.loss_fn(preds, labels)

        self.log("loss_val", loss, prog_bar=True)
        return loss

    # =========================
    # OPTIMIZER
    # =========================
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)