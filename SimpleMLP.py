import os
import re
import copy
import torch
import pandas as pd
import numpy as np
import random
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

SEED       = 1
EMBED_DIM  = 24
BATCH_SIZE = 512
EPOCHS     = 150
LR         = 0.001

CLASS_HNT = 0
CLASS_NB  = 1
CLASS_HT  = 2
N_CLASSES = 3

DATA_DIR = "../results"
TRAIN_FILES = [
    "500.perlbench_r", "502.gcc_r", "505.mcf_r", "507.cactuBSSN_r",
    "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r",
    "523.xalancbmk_r", "525.x264_r", "526.blender_r", "527.cam4_r",
    "531.deepsjeng_r", "538.imagick_r", "541.leela_r", "544.nab_r", "554.roms_r",
]
TEST_FILES = [
    "520.omnetpp_r", "549.fotonik3d_r", "557.xz_r", "503.bwaves_r", "548.exchange2_r",
]

REG_COLS = ["reg1", "reg2", "reg3", "wreg1", "wreg2", "wreg3"]

OPC_COLS = [
    "Opcode", "t_successor_ends", "f_successor_ends", "Flag_Instr_Opcode",
    "reg1_Op", "reg2_Op", "reg3_Op",
    "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
    "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
]

ROUT_COL = "Routine_Type"
ROUTINE_TYPE_MAP = {"NonLeaf": 1, "Leaf": 2, "Recursive": 3}

NUM_COLS = [
    "Size",
    "Prev_Size_1", "Prev_Size_2", "Prev_Size_3", "Prev_Size_4", "Prev_Size_5",
    "Next_Size_1", "Next_Size_2", "Next_Size_3", "Next_Size_4", "Next_Size_5",
    "Offset",
    "br_is_loop_header",
    "t_dominates", "t_post_dominates", "t_is_loop_head",
    "t_is_backedge", "t_is_loop_exit", "t_has_call",
    "f_dominates", "f_post_dominates", "f_is_loop_head",
    "f_is_backedge", "f_is_loop_exit", "f_has_call",
    "Same_BBL", "taken_ubd", "fall_ubd", "taken_store", "fall_store",
]

DROP_COLUMNS = [
    "Taken", "Executed", "Regs_Read", "Regs_Write",
    "branch_bb_addr", "taken_bb_addr", "fall_bb_addr",
    "Address", "Flag_Write_PC",
]


class NeuralNetworkWithEmbeddings(nn.Module):
    def __init__(self, reg_vocab_size, opc_vocab_size, rout_vocab_size,
                 embed_dim, num_features, n_classes,
                 hidden1=512, hidden2=256, dropout=0.4):
        super().__init__()

        self.n_reg = len(REG_COLS)
        self.n_opc = len(OPC_COLS)

        self.reg_emb  = nn.Embedding(reg_vocab_size,  embed_dim, padding_idx=0)
        self.opc_emb  = nn.Embedding(opc_vocab_size,  embed_dim, padding_idx=0)
        self.rout_emb = nn.Embedding(rout_vocab_size, embed_dim, padding_idx=0)

        in_features = num_features + (self.n_reg + self.n_opc + 1) * embed_dim

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
        x_reg  = x_cat[:, :self.n_reg]
        x_opc  = x_cat[:, self.n_reg : self.n_reg + self.n_opc]
        x_rout = x_cat[:, self.n_reg + self.n_opc]

        reg_embeds = [self.reg_emb(x_reg[:, i]) for i in range(self.n_reg)]
        opc_embeds = [self.opc_emb(x_opc[:, i]) for i in range(self.n_opc)]
        rout_embed = self.rout_emb(x_rout)

        x_cat_embedded = torch.cat(reg_embeds + opc_embeds + [rout_embed], dim=1)
        x = torch.cat([x_num, x_cat_embedded], dim=1)
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir, names):
    dfs = []
    for filename in names:
        path = data_dir + '/' + filename + ".csv"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing csv: {path}")
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def add_label(df, thr=0.005):
    rate  = df['Taken'] / df['Executed']
    label = pd.Series(CLASS_NB, index=df.index)
    label[rate < thr]     = CLASS_HNT
    label[rate > 1 - thr] = CLASS_HT
    return label.astype(int)


def add_register_features(train_df, test_df):
    def get_read_context(text):
        if pd.isna(text) or str(text).strip() == "":
            return "-1", "-1", "-1", "-1", "-1", "-1"
        matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', str(text))
        result  = []
        for name, value in matches:
            result.append(name)
            result.append(value)
        while len(result) < 6:
            result.extend(["-1", "-1"])
        return result[0], result[1], result[2], result[3], result[4], result[5]

    def get_write_context(text):
        if pd.isna(text) or str(text).strip() == "":
            return "-1", "-1", "-1"
        parts = [p.strip() for p in str(text).split() if p.strip()]
        while len(parts) < 3:
            parts.append("-1")
        return parts[0], parts[1], parts[2]

    read_cols  = ['reg1', 'reg1_Op', 'reg2', 'reg2_Op', 'reg3', 'reg3_Op']
    write_cols = ['wreg1', 'wreg2', 'wreg3']

    for df in (train_df, test_df):
        df[read_cols]  = pd.DataFrame(df["Regs_Read"].apply(get_read_context).tolist(),
                                      index=df.index)
        df[write_cols] = pd.DataFrame(df["Regs_Write"].apply(get_write_context).tolist(),
                                      index=df.index)


def build_shared_vocab(train_df, cols):
    all_vals = set()
    for col in cols:
        all_vals.update(train_df[col].astype(str).unique())
    return {val: idx + 1 for idx, val in enumerate(sorted(all_vals))}


def encode_cols(df, cols, vocab):
    return np.stack([
        df[col].astype(str).map(lambda x: vocab.get(x, 0)).values
        for col in cols
    ], axis=1)


def build_features(train_df, test_df):
    add_register_features(train_df, test_df)

    train_df = train_df.drop(columns=[c for c in DROP_COLUMNS if c in train_df.columns])
    test_df  = test_df.drop(columns=[c for c in DROP_COLUMNS if c in test_df.columns])

    for col in NUM_COLS:
        train_df[col] = train_df[col].fillna(0)
        test_df[col]  = test_df[col].fillna(0)

    scaler = StandardScaler()
    scaler.fit(train_df[NUM_COLS])
    scaler.scale_[scaler.scale_ == 0] = 1.0
    X_train_num = torch.tensor(scaler.transform(train_df[NUM_COLS]), dtype=torch.float32)
    X_test_num  = torch.tensor(scaler.transform(test_df[NUM_COLS]),  dtype=torch.float32)

    for col in REG_COLS + OPC_COLS:
        train_df[col] = train_df[col].fillna("-1").astype(str)
        test_df[col]  = test_df[col].fillna("-1").astype(str)

    reg_vocab = build_shared_vocab(train_df, REG_COLS)
    opc_vocab = build_shared_vocab(train_df, OPC_COLS)

    train_df[ROUT_COL] = train_df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0).astype(int)
    test_df[ROUT_COL]  = test_df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0).astype(int)

    X_train_cat = torch.tensor(np.hstack([
        encode_cols(train_df, REG_COLS, reg_vocab),
        encode_cols(train_df, OPC_COLS, opc_vocab),
        train_df[ROUT_COL].values.reshape(-1, 1),
    ]), dtype=torch.long)

    X_test_cat = torch.tensor(np.hstack([
        encode_cols(test_df, REG_COLS, reg_vocab),
        encode_cols(test_df, OPC_COLS, opc_vocab),
        test_df[ROUT_COL].values.reshape(-1, 1),
    ]), dtype=torch.long)

    y_train = torch.tensor(train_df["y"].values, dtype=torch.long)
    y_test  = torch.tensor(test_df["y"].values,  dtype=torch.long)

    reg_vocab_size  = len(reg_vocab) + 1
    opc_vocab_size  = len(opc_vocab) + 1
    rout_vocab_size = 4

    return (X_train_num, X_train_cat, y_train,
            X_test_num,  X_test_cat,  y_test,
            reg_vocab_size, opc_vocab_size, rout_vocab_size)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n_samples = 0.0, 0
    for x_num, x_cat, yb in loader:
        x_num, x_cat, yb = x_num.to(device), x_cat.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_num, x_cat), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_num.size(0)
        n_samples  += x_num.size(0)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x_num, x_cat, yb in loader:
        all_logits.append(model(x_num.to(device), x_cat.to(device)).cpu())
        all_labels.append(yb)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(dim=1)
    acc      = (preds == labels).float().mean().item()
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return acc, macro_f1, preds, labels


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    set_seed(SEED)
    print(f"Using {device} device")

    train_df = load_data(DATA_DIR, TRAIN_FILES)
    test_df  = load_data(DATA_DIR, TEST_FILES)
    train_df['y'] = add_label(train_df)
    test_df['y']  = add_label(test_df)

    print(f"\ntrain class distribution:")
    print(train_df["y"].value_counts(normalize=True).sort_index())
    print(f"\ntest class distribution:")
    print(test_df["y"].value_counts(normalize=True).sort_index())

    (X_train_num, X_train_cat, y_train,
     X_test_num,  X_test_cat,  y_test,
     reg_vocab_size, opc_vocab_size, rout_vocab_size) = build_features(train_df, test_df)

    print(f"\nnum features: {X_train_num.shape[1]}")
    print(f"cat features: {X_train_cat.shape[1]}")
    print(f"X_train_num:  {tuple(X_train_num.shape)}")
    print(f"X_train_cat:  {tuple(X_train_cat.shape)}")
    print(f"reg vocab:    {reg_vocab_size}")
    print(f"opc vocab:    {opc_vocab_size}")
    print(f"rout vocab:   {rout_vocab_size}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(X_train_num, X_train_cat, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader  = DataLoader(TensorDataset(X_test_num,  X_test_cat,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralNetworkWithEmbeddings(
        reg_vocab_size  = reg_vocab_size,
        opc_vocab_size  = opc_vocab_size,
        rout_vocab_size = rout_vocab_size,
        embed_dim       = EMBED_DIM,
        num_features    = X_train_num.shape[1],
        n_classes       = N_CLASSES,
    ).to(device)

    print(f"\n=== Model ===")
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_counts  = torch.tensor([(y_train == c).sum().item() for c in range(N_CLASSES)],
                                  dtype=torch.float32)
    print(f"\nclass counts:  {class_counts.tolist()}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_f1, best_state, best_epoch = -1.0, None, -1
    patience, patience_counter      = 15, 0

    print(f"\n=== Training ===\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_acc, train_f1, _, _ = evaluate(model, train_loader, device)
        test_acc,  test_f1,  _, _ = evaluate(model, test_loader,  device)

        marker = ""
        if test_f1 > best_f1:
            best_f1, best_state, best_epoch = test_f1, copy.deepcopy(model.state_dict()), epoch
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        print(f"epoch {epoch:3d}  loss={train_loss:.4f}  "
              f"train: acc={train_acc:.3f} f1={train_f1:.3f}  "
              f"test: acc={test_acc:.3f} f1={test_f1:.3f}{marker}")

        if patience_counter >= patience:
            print(f"\nEarly stop at epoch {epoch}. Best: epoch {best_epoch} (test F1 {best_f1:.3f})")
            break

    model.load_state_dict(best_state)
    print(f"\nUsing best model from epoch {best_epoch} (test macro F1 = {best_f1:.3f})")

    _, _, test_preds, test_labels = evaluate(model, test_loader, device)
    print("\n=== Test set Evaluation Report ===\n")
    print(classification_report(test_labels.numpy(), test_preds.numpy(),
                                target_names=["HNT", "NB", "HT"], digits=3))
    print("=== Test set confusion matrix ===")
    print(confusion_matrix(test_labels.numpy(), test_preds.numpy()))


if __name__ == "__main__":
    main()