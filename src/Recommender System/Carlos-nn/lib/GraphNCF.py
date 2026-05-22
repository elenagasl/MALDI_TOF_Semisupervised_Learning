import torch
import torch.nn as nn
import pytorch_lightning as pl


class MaldiEncoder(nn.Module):
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


class GraphDiffusionLayer(nn.Module):
    def __init__(
        self,
        num_items,
        W_init=None,
        graph_mode="none",
        init_alpha=0.2,
        n_steps=1,
        train_alpha=True
    ):
        """
        Graph output layer.

        graph_mode:
            - "none": no graph correction
            - "fixed": W is fixed
            - "trainable": W is learned

        Formula:
            z_graph^(0) = z_base

            z_graph^(t+1) =
                (1 - alpha) * z_base + alpha * z_graph^(t) @ W.T

        We operate on logits, not probabilities.
        """

        super().__init__()

        self.num_items = num_items
        self.graph_mode = graph_mode
        self.n_steps = n_steps

        if graph_mode not in ["none", "fixed", "trainable"]:
            raise ValueError(
                "graph_mode must be one of: 'none', 'fixed', 'trainable'"
            )

        if W_init is None:
            W_init = torch.zeros(num_items, num_items).float()

        if not torch.is_tensor(W_init):
            W_init = torch.tensor(W_init).float()

        if W_init.shape != (num_items, num_items):
            raise ValueError(
                f"W_init must have shape ({num_items}, {num_items}), "
                f"got {tuple(W_init.shape)}"
            )

        W_init = self._clean_W(W_init)

        # Store normalized initial W for regularization.
        self.register_buffer("W_init", W_init.clone())

        if graph_mode == "trainable":
            # We parameterize W directly.
            # It is later cleaned/normalized in get_W().
            self.W_param = nn.Parameter(W_init.clone())
        else:
            self.register_buffer("W_fixed", W_init.clone())
            self.W_param = None

        # alpha in (0, 1), initialized around init_alpha.
        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        alpha_logit = torch.logit(torch.tensor(float(init_alpha)))

        if graph_mode == "trainable" and train_alpha:
            self.alpha_logit = nn.Parameter(alpha_logit.clone())
        else:
            self.register_buffer("alpha_logit", alpha_logit.clone())

    def _clean_W(self, W):
        """
        Clean and row-normalize W.

        - removes NaN/inf
        - keeps only non-negative weights
        - removes diagonal
        - row-normalizes
        """

        W = W.float()
        W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

        # For first experiment we keep only positive relations.
        W = torch.clamp(W, min=0.0)

        # Remove self-loops.
        eye = torch.eye(W.shape[0], device=W.device)
        W = W * (1.0 - eye)

        row_sum = W.sum(dim=1, keepdim=True)
        W = torch.where(
            row_sum > 0,
            W / (row_sum + 1e-8),
            torch.zeros_like(W)
        )

        return W

    def get_alpha(self):
        if self.graph_mode == "none":
            return torch.tensor(0.0, device=self.W_init.device)

        return torch.sigmoid(self.alpha_logit)

    def get_W(self):
        if self.graph_mode == "none":
            return torch.zeros_like(self.W_init)

        if self.graph_mode == "fixed":
            return self.W_fixed

        W = self._clean_W(self.W_param)
        return W

    def forward(self, z_base):

        if self.graph_mode == "none" or self.n_steps <= 0:
            return z_base

        W = self.get_W().to(z_base.device)
        alpha = self.get_alpha().to(z_base.device)

        z_current = z_base

        for _ in range(self.n_steps):
            z_current = (1.0 - alpha) * z_base + alpha * torch.matmul(
                z_current,
                W.t()
            )

        return z_current


class GraphNCF(pl.LightningModule):
    def __init__(
        self,
        num_feat,
        num_items,
        graph_mode="none",
        W_init=None,
        n_graph_steps=1,
        init_alpha=0.2,
        train_alpha=True,
        drug_emb_dim=32,
        hidden_dim_CF=[128, 64],
        lr=1e-3,
        lambda_graph_reg=0.01
    ):
        """
        Vectorial recommender + optional graph output layer.

        Inputs:
            maldi: [batch, num_feat]

        Outputs:
            logits per antibiotic: [batch, num_items]

        graph_mode:
            - "none": base model
            - "fixed": graph layer with fixed W
            - "trainable": graph layer with trainable W and optionally alpha

        Loss:
            masked BCEWithLogitsLoss over observed AMR labels only.
        """

        super().__init__()

        self.num_items = num_items
        self.lr = lr
        self.graph_mode = graph_mode
        self.lambda_graph_reg = lambda_graph_reg

        # =========================
        # Encoders
        # =========================
        self.maldi_encoder = MaldiEncoder(num_feat)
        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        # =========================
        # Recommender head shared across antibiotics
        # =========================
        input_dim = 64 + drug_emb_dim

        sizes = [input_dim] + list(hidden_dim_CF) + [1]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.score_net = nn.Sequential(*layers)

        # =========================
        # Graph output layer
        # =========================
        self.graph_layer = GraphDiffusionLayer(
            num_items=num_items,
            W_init=W_init,
            graph_mode=graph_mode,
            init_alpha=init_alpha,
            n_steps=n_graph_steps,
            train_alpha=train_alpha
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward_base_logits(self, maldi):
        """
        Computes base logits for all antibiotics.

        maldi_emb: [batch, 64]
        drug_emb: [num_items, drug_emb_dim]

        Output:
            z_base: [batch, num_items]
        """

        batch_size = maldi.shape[0]

        maldi_emb = self.maldi_encoder(maldi)

        drug_ids = torch.arange(
            self.num_items,
            dtype=torch.long,
            device=maldi.device
        )

        drug_emb = self.drug_embedding(drug_ids)

        maldi_expanded = maldi_emb.unsqueeze(1).expand(
            batch_size,
            self.num_items,
            maldi_emb.shape[-1]
        )

        drug_expanded = drug_emb.unsqueeze(0).expand(
            batch_size,
            self.num_items,
            drug_emb.shape[-1]
        )

        x = torch.cat([maldi_expanded, drug_expanded], dim=-1)

        x = x.reshape(batch_size * self.num_items, -1)

        logits = self.score_net(x)

        logits = logits.view(batch_size, self.num_items)

        return logits

    def forward(self, maldi):
        z_base = self.forward_base_logits(maldi)
        z_graph = self.graph_layer(z_base)
        return z_graph

    def graph_regularization_loss(self):
        """
        Keeps learned W close to W_init.

        Only applies when graph_mode == trainable.
        """

        if self.graph_mode != "trainable":
            return torch.tensor(0.0, device=self.device)

        W = self.graph_layer.get_W()
        W_init = self.graph_layer.W_init.to(W.device)

        return torch.mean((W - W_init) ** 2)

    def masked_loss(self, logits, labels, mask):

        loss_matrix = self.loss_fn(logits, labels.float())

        masked_loss = (loss_matrix * mask.float()).sum() / (
            mask.float().sum() + 1e-8
        )

        return masked_loss

    def move_batch(self, batch):
        device = self.device
        return [x.to(device) for x in batch]

    def training_step(self, batch, batch_idx):

        maldi, labels, mask = self.move_batch(batch)

        logits = self.forward(maldi)

        loss_bce = self.masked_loss(
            logits=logits,
            labels=labels,
            mask=mask
        )

        loss_reg = self.graph_regularization_loss()

        loss = loss_bce + self.lambda_graph_reg * loss_reg

        self.log("loss_tr", loss, prog_bar=True)
        self.log("loss_bce_tr", loss_bce, prog_bar=False)
        self.log("loss_graph_reg_tr", loss_reg, prog_bar=False)

        if self.graph_mode != "none":
            alpha = self.graph_layer.get_alpha()
            self.log("alpha_tr", alpha, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):

        maldi, labels, mask = self.move_batch(batch)

        logits = self.forward(maldi)

        loss_bce = self.masked_loss(
            logits=logits,
            labels=labels,
            mask=mask
        )

        loss_reg = self.graph_regularization_loss()

        loss = loss_bce + self.lambda_graph_reg * loss_reg

        self.log("loss_val", loss, prog_bar=True)
        self.log("loss_bce_val", loss_bce, prog_bar=False)
        self.log("loss_graph_reg_val", loss_reg, prog_bar=False)

        if self.graph_mode != "none":
            alpha = self.graph_layer.get_alpha()
            self.log("alpha_val", alpha, prog_bar=False)

        return loss

    def predict_proba(self, maldi):
        self.eval()

        with torch.no_grad():
            logits = self.forward(maldi)
            probs = torch.sigmoid(logits)

        return probs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)