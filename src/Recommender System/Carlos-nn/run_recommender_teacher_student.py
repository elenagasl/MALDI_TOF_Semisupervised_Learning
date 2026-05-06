import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from pytorch_lightning.callbacks import EarlyStopping

from lib.NCF import NCF
from lib.RecDataset import RecDataset


# =========================
# CONFIG
# =========================
DATA_PATH = '/export/usuarios01/egarroyo/MALDI_for_AMR_prediction/data/COMBINED_MARISMA_DRIAMS_samples.pkl'

N_SPLITS = 2
VAL_SIZE = 0.2

BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128

MAX_EPOCHS_TEACHER = 300
MAX_EPOCHS_STUDENT = 300
PATIENCE = 15

# Teacher context config:
# Teacher sees privileged AMR context during training.
TEACHER_EMPTY_CONTEXT_PROB = 0.0
TEACHER_MIN_CONTEXT_FRACTION = 0.7
TEACHER_MAX_CONTEXT_FRACTION = 1.0

# Student distillation config:
# The batch still contains AMR context, but only teacher uses it.
DISTILL_EMPTY_CONTEXT_PROB = 0.0
DISTILL_MIN_CONTEXT_FRACTION = 0.7
DISTILL_MAX_CONTEXT_FRACTION = 1.0

# Teacher oracle diagnostic context
ORACLE_CONTEXT_FRACTION = 0.5
ORACLE_SEED = 42

# Distillation strength
LAMBDA_KD = 0.5

# Student architecture
DRUG_EMB_DIM = 32
FAMILY_EMB_DIM = 8
STUDENT_HIDDEN_DIMS = [128, 64]
LR = 1e-3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", DEVICE)

torch.backends.cudnn.benchmark = True


# =========================
# LOAD DATA
# =========================
print("Loading dataset...", flush=True)
with open(DATA_PATH, 'rb') as f:
    payload = pickle.load(f)

X_all = payload["data"]
y_species_all = payload["label"]
amr_all = payload["amr"]
antibiotics = payload["antibiotics"]

species_list = np.unique(y_species_all)


# =========================
# STUDENT MODEL
# =========================
class EmSpectrumStudent(nn.Module):
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


class StudentDistiller(pl.LightningModule):
    def __init__(
        self,
        num_feat,
        num_items,
        num_families,
        teacher=None,
        lambda_kd=0.5,
        drug_emb_dim=32,
        family_emb_dim=8,
        hidden_dim_CF=[128, 64],
        lr=1e-3
    ):
        super().__init__()

        self.num_items = num_items
        self.num_families = num_families
        self.lambda_kd = lambda_kd
        self.lr = lr

        # Optional frozen teacher
        self.teacher = teacher

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        # Student inputs:
        # MALDI + target antibiotic + dummy family
        # No AMR context.
        self.embedding_s = EmSpectrumStudent(num_feat)
        self.embedding_d = nn.Embedding(num_items, drug_emb_dim)
        self.embedding_f = nn.Embedding(num_families, family_emb_dim)

        input_dim = 64 + drug_emb_dim + family_emb_dim

        sizes = [input_dim] + list(hidden_dim_CF) + [1]

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())

        self.CF_net = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss()

    def forward(self, maldi, target_drug, target_family):

        maldi_emb = self.embedding_s(maldi)
        drug_emb = self.embedding_d(target_drug.long())
        family_emb = self.embedding_f(target_family.long())

        x = torch.cat([
            maldi_emb,
            drug_emb,
            family_emb
        ], dim=-1)

        return self.CF_net(x)

    def move_batch(self, batch):
        device = self.device
        return [x.to(device) for x in batch]

    def _shared_step(self, batch, stage):

        maldi, target_drug, target_family, labels, amr_vec, mask = self.move_batch(batch)

        labels = labels.view(-1, 1).float()

        # Student prediction: no AMR context
        p_student = self.forward(
            maldi,
            target_drug,
            target_family
        )

        loss_true = self.loss_fn(p_student, labels)

        # Distillation loss:
        # teacher sees privileged AMR context.
        if self.teacher is not None and self.lambda_kd > 0:

            self.teacher.eval()

            with torch.no_grad():
                p_teacher = self.teacher(
                    maldi,
                    target_drug,
                    target_family,
                    amr_vec,
                    mask
                )

            loss_kd = self.loss_fn(p_student, p_teacher.detach())
            loss = loss_true + self.lambda_kd * loss_kd

            self.log(f"loss_{stage}", loss, prog_bar=True)
            self.log(f"loss_true_{stage}", loss_true, prog_bar=False)
            self.log(f"loss_kd_{stage}", loss_kd, prog_bar=False)

        else:
            loss = loss_true
            self.log(f"loss_{stage}", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "tr")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.lr)


# =========================
# BUILD SPECIES DATA
# =========================
def build_species_data(species):

    mask = (y_species_all == species)

    X = X_all[mask]
    amr = amr_all[mask].copy()

    # Only binary labels are allowed:
    # 0 = susceptible
    # 1 = resistant
    # everything else becomes NaN
    valid_values = (amr == 0) | (amr == 1) | np.isnan(amr)
    amr[~valid_values] = np.nan

    if len(X) < 100:
        return None

    valid_cols = []

    for j in range(amr.shape[1]):
        col = amr[:, j]
        col = col[~np.isnan(col)]

        if len(col) > 50 and len(np.unique(col)) > 1:
            valid_cols.append(j)

    if len(valid_cols) == 0:
        return None

    X = np.asarray(X)
    amr = amr[:, valid_cols]

    return X, amr


# =========================
# FOLDS
# =========================
def create_folds(n_samples, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kf.split(np.arange(n_samples)))


def split_train_val_indices(train_idx, val_size=0.2, seed=42):

    train_inner, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    return train_inner, val_idx


# =========================
# DATASET HELPERS
# =========================
def build_sample_amr_dict(amr_matrix):
    return {i: amr_matrix[i] for i in range(len(amr_matrix))}


def make_loader(
    X,
    amr,
    antibiotic_families,
    batch_size,
    shuffle,
    empty_context_prob,
    min_context_fraction,
    max_context_fraction,
    num_workers
):

    sample_ids = np.arange(len(X))
    sample_amr = build_sample_amr_dict(amr)

    dataset = RecDataset(
        maldi=X,
        sample_ids=sample_ids,
        sample_amr=sample_amr,
        antibiotic_families=antibiotic_families,
        empty_context_prob=empty_context_prob,
        min_context_fraction=min_context_fraction,
        max_context_fraction=max_context_fraction
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


# =========================
# PREDICT ALL ANTIBIOTICS - TEACHER
# =========================
def predict_teacher_all_antibiotics(
    model,
    maldi,
    amr_context,
    context_mask,
    num_items,
    batch_size=512
):

    model.eval()

    N = maldi.shape[0]
    preds_matrix = np.zeros((N, num_items), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            maldi_batch = torch.tensor(maldi[start:end]).float().to(model.device)
            amr_batch = torch.tensor(amr_context[start:end]).float().to(model.device)
            mask_batch = torch.tensor(context_mask[start:end]).float().to(model.device)

            batch_size_real = maldi_batch.shape[0]
            drug_preds = []

            for drug_id in range(num_items):

                target_drug = torch.full(
                    (batch_size_real,),
                    drug_id,
                    dtype=torch.long,
                    device=model.device
                )

                target_family = torch.zeros(
                    batch_size_real,
                    dtype=torch.long,
                    device=model.device
                )

                # Never include target antibiotic in teacher context
                amr_tmp = amr_batch.clone()
                mask_tmp = mask_batch.clone()

                amr_tmp[:, drug_id] = 0
                mask_tmp[:, drug_id] = 0

                preds = model(
                    maldi_batch,
                    target_drug,
                    target_family,
                    amr_tmp,
                    mask_tmp
                )

                drug_preds.append(preds.view(-1))

            drug_preds = torch.stack(drug_preds, dim=1)
            preds_matrix[start:end] = drug_preds.cpu().numpy()

    return preds_matrix


# =========================
# PREDICT ALL ANTIBIOTICS - STUDENT
# =========================
def predict_student_all_antibiotics(
    model,
    maldi,
    num_items,
    batch_size=512
):

    model.eval()

    N = maldi.shape[0]
    preds_matrix = np.zeros((N, num_items), dtype=np.float32)

    with torch.no_grad():

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            maldi_batch = torch.tensor(maldi[start:end]).float().to(model.device)
            batch_size_real = maldi_batch.shape[0]

            drug_preds = []

            for drug_id in range(num_items):

                target_drug = torch.full(
                    (batch_size_real,),
                    drug_id,
                    dtype=torch.long,
                    device=model.device
                )

                target_family = torch.zeros(
                    batch_size_real,
                    dtype=torch.long,
                    device=model.device
                )

                preds = model(
                    maldi_batch,
                    target_drug,
                    target_family
                )

                drug_preds.append(preds.view(-1))

            drug_preds = torch.stack(drug_preds, dim=1)
            preds_matrix[start:end] = drug_preds.cpu().numpy()

    return preds_matrix


# =========================
# ORACLE CONTEXT FOR TEACHER
# =========================
def make_oracle_context(
    amr_true,
    num_items,
    context_fraction=0.5,
    seed=42
):

    rng = np.random.default_rng(seed)

    N = amr_true.shape[0]

    amr_context = np.zeros((N, num_items), dtype=np.float32)
    context_mask = np.zeros((N, num_items), dtype=np.float32)

    for i in range(N):

        observed = np.where(~np.isnan(amr_true[i]))[0]

        if len(observed) == 0:
            continue

        n_keep = max(1, int(context_fraction * len(observed)))

        keep_idx = rng.choice(
            observed,
            size=n_keep,
            replace=False
        )

        amr_context[i, keep_idx] = amr_true[i, keep_idx]
        context_mask[i, keep_idx] = 1

    return amr_context, context_mask


# =========================
# METRICS
# =========================
def compute_metrics(y_true, preds):

    mask = ~np.isnan(y_true)

    auc_micro = np.nan
    auc_macro = np.nan

    if np.any(mask) and len(np.unique(y_true[mask])) > 1:
        auc_micro = roc_auc_score(y_true[mask], preds[mask])

    antibiotic_aucs = []

    for j in range(y_true.shape[1]):

        col = y_true[:, j]
        valid = ~np.isnan(col)

        if np.sum(valid) > 0 and len(np.unique(col[valid])) > 1:
            antibiotic_aucs.append(
                roc_auc_score(col[valid], preds[valid, j])
            )

    if len(antibiotic_aucs) > 0:
        auc_macro = np.mean(antibiotic_aucs)

    return auc_micro, auc_macro


# =========================
# TRAIN ONE SPECIES
# =========================
def train_species(species):

    print(f"\n====================")
    print(f"Species: {species}")
    print("====================", flush=True)

    data = build_species_data(species)

    if data is None:
        print("Skipping...", flush=True)
        return None, None, None

    X, amr = data

    num_items = amr.shape[1]
    n_samples = X.shape[0]

    print("Samples:", n_samples, "Items:", num_items, flush=True)

    # No family information for now
    antibiotic_families = np.zeros(num_items, dtype=int)
    num_families = 1

    folds = create_folds(n_samples, N_SPLITS)

    teacher_micro = []
    teacher_macro = []

    baseline_micro = []
    baseline_macro = []

    distilled_micro = []
    distilled_macro = []

    for fold, (train_idx, test_idx) in enumerate(folds):

        print(f"\nFold {fold}", flush=True)

        train_inner_idx, val_idx = split_train_val_indices(
            train_idx,
            val_size=VAL_SIZE,
            seed=42 + fold
        )

        X_tr, X_val, X_tst = X[train_inner_idx], X[val_idx], X[test_idx]
        amr_tr, amr_val, amr_tst = amr[train_inner_idx], amr[val_idx], amr[test_idx]

        # =========================
        # TEACHER LOADERS
        # =========================
        teacher_loader_tr = make_loader(
            X=X_tr,
            amr=amr_tr,
            antibiotic_families=antibiotic_families,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            empty_context_prob=TEACHER_EMPTY_CONTEXT_PROB,
            min_context_fraction=TEACHER_MIN_CONTEXT_FRACTION,
            max_context_fraction=TEACHER_MAX_CONTEXT_FRACTION,
            num_workers=4
        )

        teacher_loader_val = make_loader(
            X=X_val,
            amr=amr_val,
            antibiotic_families=antibiotic_families,
            batch_size=BATCH_SIZE,
            shuffle=False,
            empty_context_prob=TEACHER_EMPTY_CONTEXT_PROB,
            min_context_fraction=TEACHER_MIN_CONTEXT_FRACTION,
            max_context_fraction=TEACHER_MAX_CONTEXT_FRACTION,
            num_workers=2
        )

        # =========================
        # TRAIN TEACHER
        # =========================
        teacher = NCF(
            num_feat=X_tr.shape[1],
            num_items=num_items,
            num_families=num_families,
            amr_dropout=0.1,
            lr=LR
        )

        teacher_trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS_TEACHER,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            devices=1,
            callbacks=[
                EarlyStopping(
                    monitor="loss_val",
                    patience=PATIENCE,
                    mode="min"
                )
            ],
            logger=False,
            enable_checkpointing=False
        )

        teacher_trainer.fit(
            teacher,
            teacher_loader_tr,
            teacher_loader_val
        )

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # =========================
        # EVALUATE TEACHER ORACLE
        # =========================
        oracle_context, oracle_mask = make_oracle_context(
            amr_true=amr_tst,
            num_items=num_items,
            context_fraction=ORACLE_CONTEXT_FRACTION,
            seed=ORACLE_SEED + fold
        )

        preds_teacher = predict_teacher_all_antibiotics(
            model=teacher,
            maldi=X_tst,
            amr_context=oracle_context,
            context_mask=oracle_mask,
            num_items=num_items
        )

        micro, macro = compute_metrics(amr_tst, preds_teacher)

        if not np.isnan(micro):
            teacher_micro.append(micro)

        if not np.isnan(macro):
            teacher_macro.append(macro)

        # =========================
        # STUDENT LOADERS
        # =========================
        # The student itself will not see AMR.
        # The AMR context in the batch is only used by the frozen teacher
        # for the distillation target.
        student_loader_tr = make_loader(
            X=X_tr,
            amr=amr_tr,
            antibiotic_families=antibiotic_families,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            empty_context_prob=DISTILL_EMPTY_CONTEXT_PROB,
            min_context_fraction=DISTILL_MIN_CONTEXT_FRACTION,
            max_context_fraction=DISTILL_MAX_CONTEXT_FRACTION,
            num_workers=4
        )

        student_loader_val = make_loader(
            X=X_val,
            amr=amr_val,
            antibiotic_families=antibiotic_families,
            batch_size=BATCH_SIZE,
            shuffle=False,
            empty_context_prob=DISTILL_EMPTY_CONTEXT_PROB,
            min_context_fraction=DISTILL_MIN_CONTEXT_FRACTION,
            max_context_fraction=DISTILL_MAX_CONTEXT_FRACTION,
            num_workers=2
        )

        # =========================
        # TRAIN BASELINE STUDENT
        # =========================
        baseline_student = StudentDistiller(
            num_feat=X_tr.shape[1],
            num_items=num_items,
            num_families=num_families,
            teacher=None,
            lambda_kd=0.0,
            drug_emb_dim=DRUG_EMB_DIM,
            family_emb_dim=FAMILY_EMB_DIM,
            hidden_dim_CF=STUDENT_HIDDEN_DIMS,
            lr=LR
        )

        baseline_trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS_STUDENT,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            devices=1,
            callbacks=[
                EarlyStopping(
                    monitor="loss_val",
                    patience=PATIENCE,
                    mode="min"
                )
            ],
            logger=False,
            enable_checkpointing=False
        )

        baseline_trainer.fit(
            baseline_student,
            student_loader_tr,
            student_loader_val
        )

        preds_baseline = predict_student_all_antibiotics(
            model=baseline_student,
            maldi=X_tst,
            num_items=num_items
        )

        micro, macro = compute_metrics(amr_tst, preds_baseline)

        if not np.isnan(micro):
            baseline_micro.append(micro)

        if not np.isnan(macro):
            baseline_macro.append(macro)

        # =========================
        # TRAIN DISTILLED STUDENT
        # =========================
        distilled_student = StudentDistiller(
            num_feat=X_tr.shape[1],
            num_items=num_items,
            num_families=num_families,
            teacher=teacher,
            lambda_kd=LAMBDA_KD,
            drug_emb_dim=DRUG_EMB_DIM,
            family_emb_dim=FAMILY_EMB_DIM,
            hidden_dim_CF=STUDENT_HIDDEN_DIMS,
            lr=LR
        )

        distilled_trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS_STUDENT,
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            devices=1,
            callbacks=[
                EarlyStopping(
                    monitor="loss_val",
                    patience=PATIENCE,
                    mode="min"
                )
            ],
            logger=False,
            enable_checkpointing=False
        )

        distilled_trainer.fit(
            distilled_student,
            student_loader_tr,
            student_loader_val
        )

        preds_distilled = predict_student_all_antibiotics(
            model=distilled_student,
            maldi=X_tst,
            num_items=num_items
        )

        micro, macro = compute_metrics(amr_tst, preds_distilled)

        if not np.isnan(micro):
            distilled_micro.append(micro)

        if not np.isnan(macro):
            distilled_macro.append(macro)

    teacher_result = {
        "species": species,
        "auc_micro": np.mean(teacher_micro) if teacher_micro else np.nan,
        "auc_macro": np.mean(teacher_macro) if teacher_macro else np.nan,
        "auc_micro_std": np.std(teacher_micro) if teacher_micro else np.nan,
        "auc_macro_std": np.std(teacher_macro) if teacher_macro else np.nan
    }

    baseline_result = {
        "species": species,
        "auc_micro": np.mean(baseline_micro) if baseline_micro else np.nan,
        "auc_macro": np.mean(baseline_macro) if baseline_macro else np.nan,
        "auc_micro_std": np.std(baseline_micro) if baseline_micro else np.nan,
        "auc_macro_std": np.std(baseline_macro) if baseline_macro else np.nan
    }

    distilled_result = {
        "species": species,
        "auc_micro": np.mean(distilled_micro) if distilled_micro else np.nan,
        "auc_macro": np.mean(distilled_macro) if distilled_macro else np.nan,
        "auc_micro_std": np.std(distilled_micro) if distilled_micro else np.nan,
        "auc_macro_std": np.std(distilled_macro) if distilled_macro else np.nan
    }

    return teacher_result, baseline_result, distilled_result


# =========================
# MAIN
# =========================
teacher_results = []
baseline_results = []
distilled_results = []

for sp in species_list:

    try:
        teacher_res, baseline_res, distilled_res = train_species(sp)

        if teacher_res is not None:
            teacher_results.append(teacher_res)

        if baseline_res is not None:
            baseline_results.append(baseline_res)

        if distilled_res is not None:
            distilled_results.append(distilled_res)

    except Exception as e:
        print(f"Error in {sp}:", e, flush=True)


df_teacher = pd.DataFrame(teacher_results)
df_baseline = pd.DataFrame(baseline_results)
df_distilled = pd.DataFrame(distilled_results)

if len(df_teacher) > 0:
    df_teacher = df_teacher.sort_values(by="auc_macro", ascending=False)

if len(df_baseline) > 0:
    df_baseline = df_baseline.sort_values(by="auc_macro", ascending=False)

if len(df_distilled) > 0:
    df_distilled = df_distilled.sort_values(by="auc_macro", ascending=False)

print("\nFINAL TEACHER ORACLE RESULTS:")
print(df_teacher)

print("\nFINAL STUDENT BASELINE RESULTS:")
print(df_baseline)

print("\nFINAL STUDENT DISTILLED RESULTS:")
print(df_distilled)

df_teacher.to_csv("teacher_oracle.csv", index=False)
df_baseline.to_csv("student_baseline.csv", index=False)
df_distilled.to_csv("student_distilled.csv", index=False)

print("\nSaved:")
print("teacher_oracle.csv")
print("student_baseline.csv")
print("student_distilled.csv")