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


class PredContextNCF(pl.LightningModule):
    """
    Recommender con contexto AMR predicho desde MALDI.

    Flujo:
        MALDI -> embedding
        embedding -> global AMR logits
        global AMR probabilities -> predicted context
        MALDI embedding + target drug embedding + context embedding -> target prediction

    En training puede usar:
        - contexto predicho
        - contexto real enmascarado
        - mezcla de ambos

    En validation/test usa contexto predicho para simular inferencia real.
    """

    def __init__(
        self,
        num_feat,
        num_items,
        drug_emb_dim=32,
        hidden_dim_CF=[128, 64],
        lr=1e-3,
        lambda_context=0.3,
        real_context_prob=0.5,
        context_dropout=0.2,
        detach_pred_context=False
    ):
        super().__init__()

        self.num_items = num_items
        self.lr = lr
        self.lambda_context = lambda_context
        self.real_context_prob = real_context_prob
        self.context_dropout = context_dropout
        self.detach_pred_context = detach_pred_context

        # =========================
        # MALDI encoder
        # =========================
        self.maldi_encoder = MaldiEncoder(num_feat)

        # =========================
        # Auxiliary AMR context head
        # =========================
        self.context_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_items)
        )

        # =========================
        # Target antibiotic embedding
        # =========================
        self.drug_embedding = nn.Embedding(num_items, drug_emb_dim)

        # =========================
        # Context encoder
        # Input:
        #   predicted/real AMR values: [num_items]
        #   context mask:              [num_items]
        # Total: 2 * num_items
        # =========================
        self.context_encoder = nn.Sequential(
            nn.Linear(num_items * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU()
        )

        # =========================
        # Final recommender head
        # =========================
        input_dim = 64 + drug_emb_dim + 64

        sizes = [input_dim] + list(hidden_dim_CF) + [1]

        layers = []

        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.target_head = nn.Sequential(*layers)

        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    # ============================================================
    # Helpers
    # ============================================================
    def move_batch(self, batch):
        device = self.device
        return [x.to(device) for x in batch]

    def masked_bce_loss(self, logits, labels, mask):
        loss_matrix = self.bce_logits(logits, labels.float())

        loss = (loss_matrix * mask.float()).sum() / (
            mask.float().sum() + 1e-8
        )

        return loss

    def build_target_exclusion_mask(self, target_drug):
        """
        Crea una máscara [batch, num_items] con 0 en el target antibiotic
        y 1 en el resto.
        """

        batch_size = target_drug.shape[0]

        mask = torch.ones(
            batch_size,
            self.num_items,
            device=target_drug.device
        )

        mask[
            torch.arange(batch_size, device=target_drug.device),
            target_drug.long()
        ] = 0.0

        return mask

    def random_context_mask(self, observed_mask, target_drug):
        """
        Construye una máscara para contexto real durante training.

        Parte de los antibióticos observados, elimina target
        y aplica dropout aleatorio de contexto.
        """

        visible_mask = observed_mask.float().clone()

        # eliminar target
        target_exclusion_mask = self.build_target_exclusion_mask(target_drug)
        visible_mask = visible_mask * target_exclusion_mask

        if self.context_dropout > 0:
            keep = torch.rand_like(visible_mask) > self.context_dropout
            visible_mask = visible_mask * keep.float()

        return visible_mask

    def make_context(
        self,
        context_logits,
        amr_labels,
        amr_mask,
        target_drug,
        use_real_context
    ):
        """
        Devuelve:
            context_values: [batch, num_items]
            context_mask:   [batch, num_items]

        Si use_real_context=True:
            usa AMR real observado con masking.
        Si use_real_context=False:
            usa AMR predicho desde MALDI.

        En ambos casos elimina el target antibiotic.
        """

        target_exclusion_mask = self.build_target_exclusion_mask(target_drug)

        if use_real_context:
            # Contexto oracle-like durante training, pero sin target.
            context_mask = self.random_context_mask(
                observed_mask=amr_mask,
                target_drug=target_drug
            )

            context_values = amr_labels.float() * context_mask

        else:
            # Contexto predicho, usado en validation/test.
            pred_probs = torch.sigmoid(context_logits)

            if self.detach_pred_context:
                pred_probs = pred_probs.detach()

            confidence = torch.abs(pred_probs - 0.5) * 2.0

            context_mask = confidence * target_exclusion_mask
            context_values = pred_probs * context_mask

            context_values = pred_probs * context_mask

        return context_values, context_mask

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        maldi,
        target_drug,
        amr_labels=None,
        amr_mask=None,
        force_pred_context=True
    ):
        """
        Si force_pred_context=True:
            usa contexto predicho.
            Esto es lo que queremos en validation/test.

        Si force_pred_context=False:
            durante training puede mezclar contexto real y predicho.
        """

        maldi_emb = self.maldi_encoder(maldi)

        context_logits = self.context_head(maldi_emb)

        if force_pred_context:
            use_real_context = False
        else:
            use_real_context = (
                torch.rand(1, device=maldi.device).item()
                < self.real_context_prob
            )

        context_values, context_mask = self.make_context(
            context_logits=context_logits,
            amr_labels=amr_labels,
            amr_mask=amr_mask,
            target_drug=target_drug,
            use_real_context=use_real_context
        )

        context_input = torch.cat(
            [context_values, context_mask],
            dim=-1
        )

        context_emb = self.context_encoder(context_input)

        drug_emb = self.drug_embedding(target_drug.long())

        final_input = torch.cat(
            [maldi_emb, drug_emb, context_emb],
            dim=-1
        )

        target_logit = self.target_head(final_input)

        return target_logit, context_logits

    # ============================================================
    # Training / validation
    # ============================================================
    def training_step(self, batch, batch_idx):
        maldi, target_drug, labels, amr_labels, amr_mask = self.move_batch(batch)

        labels = labels.view(-1, 1).float()

        # Durante training: mezcla contexto real enmascarado y contexto predicho
        target_logit, context_logits = self.forward(
            maldi=maldi,
            target_drug=target_drug,
            amr_labels=amr_labels,
            amr_mask=amr_mask,
            force_pred_context=False
        )

        loss_target = self.bce_logits(
            target_logit,
            labels
        ).mean()

        loss_context = self.masked_bce_loss(
            logits=context_logits,
            labels=amr_labels,
            mask=amr_mask
        )

        loss = loss_target + self.lambda_context * loss_context

        self.log("loss_tr", loss, prog_bar=True)
        self.log("loss_target_tr", loss_target, prog_bar=False)
        self.log("loss_context_tr", loss_context, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        maldi, target_drug, labels, amr_labels, amr_mask = self.move_batch(batch)

        labels = labels.view(-1, 1).float()

        # En validation usamos SIEMPRE contexto predicho.
        # Esto simula inferencia real.
        target_logit, context_logits = self.forward(
            maldi=maldi,
            target_drug=target_drug,
            amr_labels=amr_labels,
            amr_mask=amr_mask,
            force_pred_context=True
        )

        loss_target = self.bce_logits(
            target_logit,
            labels
        ).mean()

        loss_context = self.masked_bce_loss(
            logits=context_logits,
            labels=amr_labels,
            mask=amr_mask
        )

        loss = loss_target + self.lambda_context * loss_context

        self.log("loss_val", loss, prog_bar=True)
        self.log("loss_target_val", loss_target, prog_bar=False)
        self.log("loss_context_val", loss_context, prog_bar=False)

        return loss

    def predict_target_proba(self, maldi, target_drug):
        self.eval()

        with torch.no_grad():
            target_logit, _ = self.forward(
                maldi=maldi,
                target_drug=target_drug,
                amr_labels=None,
                amr_mask=None,
                force_pred_context=True
            )

            proba = torch.sigmoid(target_logit)

        return proba

    def predict_all_antibiotics(self, maldi):
        """
        Predice todos los antibióticos para un batch de muestras.

        Para cada target antibiotic:
            - usa el mismo predicted AMR context
            - elimina el target del contexto
        """

        self.eval()

        batch_size = maldi.shape[0]
        all_preds = []

        with torch.no_grad():
            for drug_id in range(self.num_items):
                target_drug = torch.full(
                    (batch_size,),
                    drug_id,
                    dtype=torch.long,
                    device=maldi.device
                )

                target_logit, _ = self.forward(
                    maldi=maldi,
                    target_drug=target_drug,
                    amr_labels=None,
                    amr_mask=None,
                    force_pred_context=True
                )

                proba = torch.sigmoid(target_logit).view(-1)
                all_preds.append(proba)

        all_preds = torch.stack(all_preds, dim=1)

        return all_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)