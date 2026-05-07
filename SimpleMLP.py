import os
import re
import torch
import pandas as pd
import numpy as np
import random
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset



# CONFIGURATIONS
SEED = 1
EMBED_DIM=16
BATCH_SIZE=256
EPOCHS = 50
LR = 0.001

CLASS_HNT=0 # Label for Highly Not Taken
CLASS_NB=1  # Label for Not Biased
CLASS_HT=2  # Label for Highly Taken
N_CLASSES=3 # Number of labels

DATA_DIR="../results"
TRAIN_FILES = [
    "500.perlbench_r",
    "502.gcc_r",
    "505.mcf_r",
    "507.cactuBSSN_r",
    "508.namd_r",
    "510.parest_r",
    "511.povray_r",
    "519.lbm_r",
    "523.xalancbmk_r",
    "525.x264_r",
    "526.blender_r",
    "527.cam4_r",
    "531.deepsjeng_r",
    "538.imagick_r",
    "541.leela_r",
    "544.nab_r",
    "554.roms_r",
]
TEST_FILES = [
    "520.omnetpp_r",
    "549.fotonik3d_r",
    "557.xz_r",
    "503.bwaves_r",
    "548.exchange2_r"
]

# COLUMNS (Features)

# Register Columns store string values (Names of registers)
REGS =[
    "reg1",
    "reg2",
    "reg3",
    "wreg1",
    "wreg2",
    "wreg3",
]

# Columns that need to get dropped
DROP_COLUMNS = [
    "Taken", 
    "Executed", 
    "Regs_Read",  
    "Regs_Write", 
    "branch_bb_addr", 
    "taken_bb_addr", 
    "fall_bb_addr"
]

# Opcode columns store string values (Names of Opcodes)
OPCODE_COLS = [
    "Opcode",
    "t_successor_ends",
    "f_successor_ends",
    "Flag_Instr_Opcode",
    "reg1_Op", "reg2_Op", "reg3_Op", "wreg1_Op", "wreg2_Op", "wreg3_Op",
    "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
    "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
]

# Hex addresses 
HEX_ADDR_COLS = [    "Address",
    "Flag_Write_PC",]

# Routine_Type has a known fixed vocabulary
ROUTINE_TYPE_COL = "Routine_Type"

# Size in bytes columns
SIZE_COLS = [
    "Size",
    "Prev_Size_1", "Prev_Size_2", "Prev_Size_3", "Prev_Size_4", "Prev_Size_5",
    "Next_Size_1", "Next_Size_2", "Next_Size_3", "Next_Size_4", "Next_Size_5",
]

# Offset column that holds value negative or positive
OFFSET_COL = ["Offset"]

# Columns that holds values -1,0,1
BOOL_COLS= [
    "br_is_loop_header",
    "t_dominates",
    "t_post_dominates",
    "t_is_loop_head",
    "t_is_backedge",
    "t_is_loop_exit",
    "t_has_call",
    "f_dominates",
    "f_post_dominates",
    "f_is_loop_head",
    "f_is_backedge",
    "f_is_loop_exit",
    "f_has_call", 
    "Same_BBL" ,
    "taken_ubd",
    "fall_ubd", 
    "taken_store",
    "fall_store",
]

# All numerical columns
NUM_COLS = (
    SIZE_COLS
    + OFFSET_COL
    + BOOL_COLS
    + HEX_ADDR_COLS
)

class NeuralNetworkWithEmbeddings(nn.Module):
    def __init__(self, vocab_sizes, embed_dim, num_features, n_classes,
                 hidden1=256, hidden2=128, dropout=0.2):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])

        in_features = num_features + len(vocab_sizes) * embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, n_classes)
        )

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_embedded = torch.cat(embedded, dim=1)  

        x = torch.cat([x_num, x_cat_embedded], dim=1)  
        return self.net(x)
    


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# Take input the directory, filenames 
# Return their context
def load_data(data_dir:str, names:list[str]) -> pd.DataFrame:
    dfs=[]
    for filename in names:
        path = data_dir + '/' + filename + ".csv"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing csv: {path}")
        df=pd.read_csv(path)
        dfs.append(df)

    return pd.concat(dfs,ignore_index=True)

# Take input a dataframe, threshhold 
# Adds the label column at the dataframe
def add_label(df:pd.DataFrame, thr: float = 0.005) -> pd.Series:
    rate = df['Taken']/df['Executed']
    label = pd.Series(CLASS_NB, index=df.index)
    label[rate < thr]      = CLASS_HNT
    label[rate > 1-thr]      = CLASS_HT

    return label.astype(int)

# Make every value float
def add_register_features(train_df, test_df):

    def get_context(text):
        pattern = r'([a-zA-Z0-9_]+)\(([^)]+)\)'
        matches = re.findall(pattern, str(text))

        result = []

        for name, value in matches:
            result.append(name)
            result.append(value)

        while len(result) < 6:
            result.extend(["-1", "-1"])

        return result[0], result[1], result[2], result[3], result[4], result[5]
        
    cols_w = ['wreg1', 'wreg1_Op', 'wreg2', 'wreg2_Op', 'wreg3', 'wreg3_Op']
    cols = ['reg1', 'reg1_Op', 'reg2', 'reg2_Op', 'reg3', 'reg3_Op']
    for df in (train_df, test_df):
        df[cols]   = pd.DataFrame(df["Regs_Read"].apply(get_context).tolist(),
                              index=df.index)
        df[cols_w] = pd.DataFrame(df["Regs_Write"].apply(get_context).tolist(),
                              index=df.index)



def vocab_build(train_df,test_df,cat_cols):
    vocabs = {}
    for col in cat_cols:
        unique_vals = train_df[col].unique()
        vocab = {val: idx+1 for idx, val in enumerate(unique_vals)}
        vocabs[col] = vocab
    return vocabs


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Register features splitted (read, write) 
    add_register_features(train_df,test_df)

    for col in HEX_ADDR_COLS:
        train_df[col] = train_df[col].apply(lambda x: int(x,16))
        test_df[col] = test_df[col].apply(lambda x: int(x,16))

    # Not needed columns
    train_df = train_df.drop(columns = DROP_COLUMNS)
    test_df = test_df.drop(columns = DROP_COLUMNS)

    cat_cols = REGS + OPCODE_COLS + [ROUTINE_TYPE_COL]

    for col in NUM_COLS:
        train_df[col]= train_df[col].fillna(0)
        test_df[col]= test_df[col].fillna(0)
    
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("-1").astype(str)
        test_df[col] = test_df[col].fillna("-1").astype(str)
        
    scaler = StandardScaler()
    scaler.fit(train_df[NUM_COLS])
    scaler.scale_[scaler.scale_ == 0] = 1.0

    x_train_num= scaler.transform(train_df[NUM_COLS])
    x_test_num = scaler.transform(test_df[NUM_COLS])

    vocabs = vocab_build(train_df, test_df, cat_cols)
    vocab_sizes = []
    train_cat_indices = []
    test_cat_indices  = []

    for col in cat_cols:
        vocab = vocabs[col]
        vocab_sizes.append(len(vocab) + 1)  # +1 for unknown (index 0)

        train_cat_indices.append(
            train_df[col].map(lambda x: vocab.get(x, 0)).values
        )
        test_cat_indices.append(
            test_df[col].map(lambda x: vocab.get(x, 0)).values
        )

    # Shape: (n_samples, n_cat_cols)
    X_train_cat = torch.tensor(np.stack(train_cat_indices, axis=1), dtype=torch.long)
    X_test_cat  = torch.tensor(np.stack(test_cat_indices,  axis=1), dtype=torch.long)

    y_train = torch.tensor(train_df["y"].values, dtype=torch.long)
    y_test  = torch.tensor(test_df["y"].values,  dtype=torch.long)

    return x_train_num, X_train_cat, y_train, x_test_num, X_test_cat, y_test, vocab_sizes



    
    
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x_num, x_cat, yb in loader:
        x_num, x_cat, yb = x_num.to(device), x_cat.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_num.size(0)
        n_samples  += x_num.size(0)

    return total_loss/n_samples
    
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x_num, x_cat, yb in loader:
        x_num, x_cat = x_num.to(device), x_cat.to(device)
        logits = model(x_num, x_cat).cpu()
        all_logits.append(logits)
        all_labels.append(yb)
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=1)
    acc = (preds == labels.int()).float().mean().item()
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    

    return acc,macro_f1, preds, labels

    
def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available()  else "cpu"
    set_seed(SEED)
    print(f"Using {device} device")
    # Load data 
    train_df= load_data(DATA_DIR,TRAIN_FILES)
    test_df= load_data(DATA_DIR,TEST_FILES)

    # Add Label column
    train_df['y'] = add_label(train_df)
    test_df['y'] = add_label(test_df)

    # Sanity check
    print(f"\ntrain class distribution:")
    print(train_df["y"].value_counts(normalize=True).sort_index())
    print(f"\ntest class distribution:")
    print(test_df["y"].value_counts(normalize=True).sort_index())

    X_train_num, X_train_cat, y_train,X_test_num, X_test_cat, y_test, vocab_sizes = build_features(train_df, test_df)

    print(f"Numeric features   : {X_train_num.shape[1]}")
    print(f"Categorical columns: {X_train_cat.shape[1]}")
    print(f"Embed dim          : {EMBED_DIM}")
    print(f"Total embed output : {X_train_cat.shape[1] * EMBED_DIM}")
    print(f"Total input to MLP : {X_train_num.shape[1] + X_train_cat.shape[1] * EMBED_DIM}")

    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_test_num  = torch.tensor(X_test_num,  dtype=torch.float32)

    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
    test_dataset  = TensorDataset(X_test_num,  X_test_cat,  y_test)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralNetworkWithEmbeddings(
        vocab_sizes  = vocab_sizes,
        embed_dim    = EMBED_DIM,
        num_features = X_train_num.shape[1],
        n_classes    = N_CLASSES
    ).to(device)
    print("\n=== Model ===\n")
    print(model)

    class_counts = torch.tensor([(y_train == c).sum().item() for c in range(N_CLASSES)], dtype=torch.float32)
    class_weights = (class_counts.sum() / (N_CLASSES * class_counts)).to(device)
    print(f"class counts : {class_counts.tolist()}")
    print(f"class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print("\n=== Training ===\n")

    for epoch in range(1,EPOCHS+1):
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
        train_acc, train_f1, _,_= evaluate(model,train_loader,device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:3d}  loss={train_loss:.4f}  "
            f"train: acc={train_acc:.3f} f1={train_f1:.3f}  "
            f"lr={current_lr:.6f}")
    

    print(f"\nFinished training for {EPOCHS} epochs.")
    test_acc, test_f1,test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"\nFinal test: acc={test_acc:.3f} f1={test_f1:.3f}")

    class_names = ["HNT", "NB", "HT"]

    print("\n=== Test set Evaluation Report ===\n")
    print(classification_report(test_labels.numpy(), test_preds.numpy(), target_names=class_names, digits=3))

    print("=== Test set confusion matrix ===")
    print(confusion_matrix(test_labels.numpy(), test_preds.numpy()))





if __name__=="__main__":
    main()