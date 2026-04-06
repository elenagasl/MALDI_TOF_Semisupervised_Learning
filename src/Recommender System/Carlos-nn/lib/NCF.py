import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchmetrics import AUROC
from lib.MaldiTransformer import MaldiTransformer

class Em_Spectrum(nn.Module):
    """ Embedding layer for the samples (MADLI-TOF spectrum)

        Args:
            input_dim (int): Dimension of the spectrum
            hidden_dim (int): Hidden dimension of the spectrum embedding
            output_dim (int): Dimension of the spectrum embedding
    """
    def __init__(self, input_dim, output_dim, hidden_dim = [64], architecture = "MLP"):
        super().__init__()
        self.architecture = architecture
        sizes = [input_dim] + list(map(int, hidden_dim)) + [output_dim]
        p = 0.2 # dropout probability
        if architecture == "MLP":
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p))
                layers.append(nn.LayerNorm(sizes[i+1]))
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
            self.net = nn.Sequential(*layers)
        elif architecture == "CNN":
            # Define the 1D CNN layer
            self.cnn = nn.Conv1d(in_channels=1, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
            # Define a global average pooling layer
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            # Define the final linear layer
        elif architecture == "Transformer":
            print("Transformer architecture not implemented yet")
            self.transformer = MaldiTransformer(hidden_dim[0], output_dim, n_classes=64,
                    n_heads=8,
                    dropout=0.2,
                    p=0.2,
                    clf=False,
                    clf_train_p=1 / 100,
                    lr=0.0005,
                    weight_decay=0,
                    lr_decay_factor=1,
                    warmup_steps=2500,
                    lmbda = 1,)
        else:
            print("Architecture not implemented")
            return

    def forward(self, x):
        x = x.float()  # Ensure x is a float tensor
        if self.architecture == "MLP":
            return self.net(x)
        elif self.architecture == "CNN":
            # Add an extra dimension to x for CNN architecture
            x = x.unsqueeze(1)  # Transform x to shape: batch_size x 1 x length
            x = self.cnn(x)
            x = self.global_pool(x)
            return x.view(x.size(0), -1)  # Flatten the output
        else:
            return

class Embedding(nn.Module):
    """ Embedding layer

        Args:
            num_embeddings (int): Number of unique lements
            dim_embeddings (int): Dimension of the embedding
    """
    def __init__(self, num_embeddings, dim_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, dim_embeddings)

    def forward(self, x):
        x = self.embedding(x.long())
        return x

# NCF class
class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_feats (int): Number of features in samples.
            num_items (int): Number of unique items.
            num_meta (int): Number of unique metadata.
            
            samples (np.array): Array with the samples. We are considering these are features instead of ids.
            item_id (np.array): Array with the item ids
            ratings (np.array): Array with the ratings
            all_itemIds (np.array): Array with all the unique item ids
            embedding_dim_samples (int): Dimension of the spectrum embedding
            embedding_dim_items (int): Dimension of the item embedding
            hidden_dim_samples (int): Hidden dimension of the spectrum embedding
    """

    def __init__(self, num_feat, num_items, num_meta = None, sample_encoder = 'MLP', embedding_dim_samples=15, embedding_dim_items=15, embedding_dim_metadata=15, hidden_dim_samples=[64], hidden_dim_CF=[64,32]):
        super().__init__()
        #TODO: Add the option of not having metadata
        num_meta = num_meta if num_meta is not None else 0
        # Embedding layers
        self.embedding_s = Em_Spectrum(num_feat, embedding_dim_samples, hidden_dim_samples, architecture = sample_encoder)
        self.embedding_d = Embedding(num_items, embedding_dim_items)
        self.embedding_m = Embedding(num_meta, embedding_dim_metadata)
        
        # Dense layers for the collaborative filtering
        sizes = [embedding_dim_samples+embedding_dim_items+embedding_dim_metadata] + list(map(int, hidden_dim_CF)) + [1]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Sigmoid())
        self.CF_net = nn.Sequential(*layers)
        
        # Metrics
        self.acc_tr = Accuracy(task="binary")
        self.acc_val = Accuracy(task="binary")
        self.acc_tst = Accuracy(task="binary")
        self.auc_tr = AUROC(task='binary')
        self.auc_val = AUROC(task='binary')
        self.auc_tst = AUROC(task='binary')

    def forward(self, user_input, meta_input, item_input):

        # Pass through embedding layers
        user_embedded = self.embedding_s(user_input)
        meta_embedded = self.embedding_m(meta_input)
        item_embedded = self.embedding_d(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, meta_embedded, item_embedded], dim=-1)

        # Pass through the collaborative filtering layers
        pred = self.CF_net(vector)

        return pred

    def training_step(self, batch, batch_idx):
        user_input, meta_input, item_input, labels = batch
        predicted_labels = self.forward(user_input, meta_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.acc_tr(predicted_labels, labels.view(-1, 1).int())
        self.auc_tr.update(predicted_labels, labels.view(-1, 1).float())
        self.log('loss_tr', loss)
        return loss

    def on_train_epoch_end(self):
        values = {"acc_tr": self.acc_tr, "auc_tr": self.auc_tr.compute()}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        self.auc_tr.reset()

    def validation_step(self, batch, batch_idx):
        user_input, meta_input, item_input, labels = batch
        predicted_labels = self.forward(user_input, meta_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.acc_val(predicted_labels, labels.view(-1, 1).int())
        self.auc_val.update(predicted_labels, labels.view(-1, 1).float())
        self.log('loss_val', loss)
        return loss
    
    def on_validation_epoch_end(self):
        values = {"acc_val": self.acc_val, "auc_val": self.auc_val.compute()}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        self.auc_val.reset()

    def test_step(self, batch, batch_idx):
        user_input, meta_input, item_input, labels = batch
        predicted_labels = self.forward(user_input, meta_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        self.acc_tst(predicted_labels, labels.view(-1, 1).int())
        self.auc_tst.update(predicted_labels, labels.view(-1, 1).float())
        self.log('loss_tst', loss)
        return loss

    def on_test_epoch_end(self):
        values = {"acc_tst": self.acc_tst, "auc_tst": self.auc_tst.compute()}
        self.log_dict(values, prog_bar=True, on_epoch=True)
        self.auc_tst.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['metrics'] = self.log_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())