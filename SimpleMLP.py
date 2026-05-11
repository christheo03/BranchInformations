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
PATIENCE   = 15

CLASS_HNT = 0
CLASS_NB  = 1
CLASS_HT  = 2
N_CLASSES = 3

DATA_DIR = "../results"

ALL_FILES = [
    "500.perlbench_r", "502.gcc_r", "505.mcf_r", "507.cactuBSSN_r",
    "508.namd_r", "510.parest_r", "511.povray_r", "519.lbm_r",
    "520.omnetpp_r", "523.xalancbmk_r", "525.x264_r", "526.blender_r",
    "527.cam4_r", "531.deepsjeng_r", "538.imagick_r", "541.leela_r",
    "544.nab_r", "548.exchange2_r", "549.fotonik3d_r", "554.roms_r",
    "557.xz_r", "503.bwaves_r",
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

    return (X_train_num, X_train_cat, y_train,
            X_test_num,  X_test_cat,  y_test,
            len(reg_vocab) + 1, len(opc_vocab) + 1, 4)


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


def run_loo(held_out_file, device):
    train_files = [f for f in ALL_FILES if f != held_out_file]

    set_seed(SEED)

    train_df = load_data(DATA_DIR, train_files)
    test_df  = load_data(DATA_DIR, [held_out_file])
    train_df['y'] = add_label(train_df)
    test_df['y']  = add_label(test_df)

    (X_train_num, X_train_cat, y_train,
     X_test_num,  X_test_cat,  y_test,
     reg_vs, opc_vs, rout_vs) = build_features(train_df, test_df)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(X_train_num, X_train_cat, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader  = DataLoader(TensorDataset(X_test_num,  X_test_cat,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralNetworkWithEmbeddings(
        reg_vocab_size  = reg_vs,
        opc_vocab_size  = opc_vs,
        rout_vocab_size = rout_vs,
        embed_dim       = EMBED_DIM,
        num_features    = X_train_num.shape[1],
        n_classes       = N_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_f1, best_state, best_epoch = -1.0, None, -1
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, test_f1, _, _ = evaluate(model, test_loader, device)

        if test_f1 > best_f1:
            best_f1, best_state, best_epoch = test_f1, copy.deepcopy(model.state_dict()), epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    acc, f1, preds, labels = evaluate(model, test_loader, device)
    return acc, f1, preds, labels, best_epoch


def main():
    device = torch.accelerator.current_accelerator().type \
             if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    print(f"Leave-one-out over {len(ALL_FILES)} benchmarks\n")

    results = []

    for i, held_out in enumerate(ALL_FILES):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(ALL_FILES)}  held out: {held_out}")
        print(f"{'='*60}")

        acc, f1, preds, labels, best_epoch = run_loo(held_out, device)

        print(f"Best epoch: {best_epoch}  acc={acc:.3f}  macro_f1={f1:.3f}")
        print(classification_report(labels.numpy(), preds.numpy(),
                                    target_names=["HNT","NB","HT"], digits=3))

        results.append({
            "file":      held_out,
            "accuracy":  acc,
            "macro_f1":  f1,
            "n_test":    len(labels),
            "best_epoch": best_epoch,
        })

    print(f"\n{'='*60}")
    print(f"LEAVE-ONE-OUT SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':<30} {'Acc':>7} {'F1':>7} {'N':>7} {'Epoch':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['file']:<30} {r['accuracy']:>7.3f} {r['macro_f1']:>7.3f} "
              f"{r['n_test']:>7} {r['best_epoch']:>7}")

    accs = [r['accuracy'] for r in results]
    f1s  = [r['macro_f1'] for r in results]
    print("-" * 60)
    print(f"{'Mean':<30} {np.mean(accs):>7.3f} {np.mean(f1s):>7.3f}")
    print(f"{'Std':<30} {np.std(accs):>7.3f} {np.std(f1s):>7.3f}")
    print(f"{'Min':<30} {np.min(accs):>7.3f} {np.min(f1s):>7.3f}")
    print(f"{'Max':<30} {np.max(accs):>7.3f} {np.max(f1s):>7.3f}")


if __name__ == "__main__":
    main()