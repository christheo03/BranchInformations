
from ml_configs import torch, nn, REG_COLS, OPC_COLS, f1_score,random,np


class NeuralNetworkWithEmbeddings(nn.Module):
    def __init__(
        self,
        reg_vocab_size,
        opc_vocab_size,
        embed_dim,
        num_features,
        n_classes,
        hidden1=512,
        hidden2=256,
        dropout=0.4,
    ):
        super().__init__()

        self.n_reg = len(REG_COLS)
        self.n_opc = len(OPC_COLS)

        self.reg_emb = nn.Embedding(reg_vocab_size, embed_dim, padding_idx=0)
        self.opc_emb = nn.Embedding(opc_vocab_size, embed_dim, padding_idx=0)
        in_features = num_features + (self.n_reg + self.n_opc) * embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x_num, x_cat):
        x_reg = x_cat[:, :self.n_reg]
        x_opc = x_cat[:, self.n_reg:self.n_reg + self.n_opc]
        reg_embeds = [
            self.reg_emb(x_reg[:, i])
            for i in range(self.n_reg)
        ]

        opc_embeds = [
            self.opc_emb(x_opc[:, i])
            for i in range(self.n_opc)
        ]

        x_cat_embedded = torch.cat(
            reg_embeds + opc_embeds,
            dim=1
        )

        x = torch.cat([x_num, x_cat_embedded], dim=1)

        return self.net(x)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for x_num, x_cat, yb in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        logits = model(x_num, x_cat).cpu()

        all_logits.append(logits)
        all_labels.append(yb)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    preds = logits.argmax(dim=1)

    acc = (preds == labels).float().mean().item()

    macro_f1 = f1_score(
        labels.numpy(),
        preds.numpy(),
        average="macro",
    )

    return acc, macro_f1, preds, labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False