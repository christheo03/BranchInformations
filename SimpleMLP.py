import os
import re
import torch
import pandas as pd
import numpy as np
from torch import nn

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset



# CONFIGURATIONS

BATCH_SIZE=256
EPOCHS = 20
LR = 0.001

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
HEX_ADDR_COLS = ["Address", "Flag_Write_PC"]

# Routine_Type has a known fixed vocabulary
ROUTINE_TYPE_COL = "Routine_Type"
ROUTINE_TYPE_MAP = {"NonLeaf": 0, "Leaf": 1, "Recursive": 2}

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

class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features,hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2,1)
        )

    def forward(self,x):
        logits = self.net(x)
        return logits.squeeze(-1)

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
def add_lable(df:pd.DataFrame, thr: float = 0.05) -> pd.Series:
    rate = df['Taken']/df['Executed']
    return ((rate < thr) | (rate > 1 - thr)).astype(int)

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

# Input a group and converts it to 
def encode_group(train_df, test_df , group):
    le = LabelEncoder()
    union = pd.concat([train_df[c].astype(str) for c in group], ignore_index=True)
    le.fit(union)
    known =set(le.classes_)

    for col in group:
        train_vals = train_df[col].astype(str)
        test_vals  = test_df[col].astype(str)
        test_vals  = test_vals.where(test_vals.isin(known), other=le.classes_[0])
        train_df[col] = le.transform(train_vals)
        test_df[col]  = le.transform(test_vals)

    return le

def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Register features splitted (read, write) 
    add_register_features(train_df,test_df)
    if "Address" in train_df.columns and "Flag_Write_PC" in train_df.columns:
        for col in ["Address","Flag_Write_PC"]:
            train_df[col] = train_df[col].apply(lambda x: int(x,16))
            test_df[col] = test_df[col].apply(lambda x: int(x,16))

    # Not needed columns
    train_df = train_df.drop(columns = DROP_COLUMNS)
    test_df = test_df.drop(columns = DROP_COLUMNS)


    reg_encoder=encode_group(train_df,test_df,REGS)
    opcode_encoder=encode_group(train_df,test_df,OPCODE_COLS)
    

    # Encode Routine_Type with its fixed known vocabulary
    train_df[ROUTINE_TYPE_COL] = train_df[ROUTINE_TYPE_COL].map(ROUTINE_TYPE_MAP)
    test_df[ROUTINE_TYPE_COL]  = test_df[ROUTINE_TYPE_COL].map(ROUTINE_TYPE_MAP)

    # Fill any missing values in numeric columns with 0
    for col in NUM_COLS:
        train_df[col] = train_df[col].fillna(0)
        test_df[col]  = test_df[col].fillna(0)
        
    scaler = StandardScaler()
    scaler.fit(train_df[NUM_COLS])
    scaler.scale_[scaler.scale_ == 0] = 1.0
    train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])
    test_df[NUM_COLS] = scaler.transform(test_df[NUM_COLS])

    # Only feature columns
    feature_cols= [c for c in train_df.columns if c != 'y']

    # Tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["y"].values,         dtype=torch.float32)
    X_test  = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test  = torch.tensor(test_df["y"].values,          dtype=torch.float32)

    return X_train, y_train, X_test, y_test, feature_cols, reg_encoder, opcode_encoder
    
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb, yb =xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss+= loss.item() * xb.size(0)
        n_samples += xb.size(0)

    return total_loss/n_samples
    
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu()
        all_logits.append(logits)
        all_labels.append(yb)
    
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits)>0.5).int()
    acc = (preds == labels.int()).float().mean().item()

    return acc, preds, labels

    
def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available()  else "cpu"
    print(f"Using {device} device")
    # Load data 
    train_df= load_data(DATA_DIR,TRAIN_FILES)
    test_df= load_data(DATA_DIR,TEST_FILES)

    # Add Label column
    train_df['y'] = add_lable(train_df)
    test_df['y'] = add_lable(test_df)

    # Sanity check
    print("\ntrain label distribution:")
    print(train_df["y"].value_counts())
    print(f"train biased rate: {train_df['y'].mean():.3f}")
    print(f"test  biased rate: {test_df ['y'].mean():.3f}")

    
    X_train, y_train, X_test, y_test, feature_cols, reg_enc, opc_enc = build_features(train_df,test_df)
    nan_cols = []
    for i, col in enumerate(feature_cols):
        if torch.isnan(X_train[:, i]).any():
            n_nan = torch.isnan(X_train[:, i]).sum().item()
            nan_cols.append((col, n_nan))
            print(f"NaN in column {col}: {n_nan} rows")

    if not nan_cols:
        print("No NaN columns found in tensor — must be from scaler scale_=0")
    print("\n=== Returned tensors ===")
    print(f"feature count: {len(feature_cols)}")
    print(f"X_train: shape={tuple(X_train.shape)}, dtype={X_train.dtype}")
    print(f"X_test:  shape={tuple(X_test.shape)}, dtype={X_test.dtype}")
    print(f"y_train: shape={tuple(y_train.shape)}, dtype={y_train.dtype}")
    print(f"y_test:  shape={tuple(y_test.shape)}, dtype={y_test.dtype}")

    print("\n=== Register encoder ===")
    print(f"vocabulary size: {len(reg_enc.classes_)}")
    for i, name in enumerate(reg_enc.classes_):
        print(f"  {i:3d} -> {name}")

    print("\n=== Opcode encoder ===")
    print(f"vocabulary size: {len(opc_enc.classes_)}")
    for i, name in enumerate(opc_enc.classes_):
        print(f"  {i:3d} -> {name}")

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test,y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    in_features= X_train.shape[1]

    model = NeuralNetwork(in_features).to(device)
    print("\n=== Model ===\n")
    print(model)

    pos_count = (y_train == 1).sum().float()
    neg_count = (y_train == 0).sum().float()
    pos_weight = (neg_count/pos_count).to(device)
    print(f"\npos_weight: {pos_weight.item():.3f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\n=== Training ===\n")
    for epoch in range(1,EPOCHS+1):
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
        train_acc, _, _ = evaluate(model,train_loader,device)
        test_acc, _, _ =evaluate(model,test_loader,device)
        print(f"epoch {epoch:3d}  loss={train_loss:.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")


    _, test_preds, test_labels = evaluate(model, test_loader, device)
    print("\n=== Test set Evaluation Report ===\n")
    print(classification_report(test_labels.numpy(),test_preds.numpy(),digits=3))
    print("=== Test set confusion matrix ===")
    print(confusion_matrix(test_labels.numpy(), test_preds.numpy()))





if __name__=="__main__":
    main()