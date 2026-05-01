import os
import re
import torch
import pandas as pd
from torch import nn

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset



BATCH_SIZE=256
EPOCHS = 20
LR = 0.001

DATA_DIR="../results"
TRAIN_FILES = [
    # "500.perlbench_r",
    # "502.gcc_r",
    "505.mcf_r",
    # "507.cactuBSSN_r",
    # "508.namd_r",
    # "510.parest_r",
    # "511.povray_r",
    "519.lbm_r",
]
TEST_FILES = [
    "520.omnetpp_r",
]

CATEGORICAL_COLS = [
    "Opcode",
    "t_successor_ends",
    "f_successor_ends",
    "Routine_Type",
    "reg1",  "reg1_Op",
    "reg2",  "reg2_Op",
    "reg3",  "reg3_Op",
    "wreg1", "wreg1_Op",
    "wreg2", "wreg2_Op",
    "wreg3", "wreg3_Op",
]

DROP_COLUMNS = ["Taken", "Executed", "Regs_Read",  "Regs_Write"]

OPCODE_COLS = [
    "Flag_Instr_Opcode",
    "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
    "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
    ]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

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
        matches = re.findall(pattern, text)

        result = []

        for name, value in matches:
            result.append(name)
            result.append(value.lower())

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


    for col in  OPCODE_COLS:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # Only feature columns
    feature_cols= [c for c in train_df.columns if c != 'y']

    # Union the columns
    cat_cols = list(set(CATEGORICAL_COLS) | set(OPCODE_COLS))
    num_cols = [c for c in feature_cols if c not in cat_cols]

    GROUPS = [
        ["reg1", "reg2", "reg3", "wreg1", "wreg2", "wreg3"],# Register names (str)
        ["Opcode","reg1_Op", "reg2_Op", "reg3_Op", "wreg1_Op", "wreg2_Op", "wreg3_Op"], # Opcodes  (str)
        ["Flag_Instr_Opcode",
         "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
         "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5"], # Opcodes (int)
        ]
    
    grouped = set()

    # Encoder for each group
    for group in GROUPS:
        le = LabelEncoder()

        group_union = pd.concat(
            [train_df[c].astype(str) for c in group],
            ignore_index=True,
        )
        le.fit(group_union)
        known = set(le.classes_)

        for col in group:
            train_vals = train_df[col].astype(str)
            test_vals = test_df[col].astype(str)
            test_vals = test_vals.where(test_vals.isin(known), other = le.classes_[0])

            train_df[col] = le.transform(train_vals)
            test_df[col] = le.transform(test_vals)
            grouped.add(col)

    for col in cat_cols:
        if col in grouped:
            continue
        le = LabelEncoder()
        train_vals= train_df[col].astype(str)
        test_vals = test_df[col].astype(str)

        le.fit(train_vals)
        known = set(le.classes_)
        test_vals = test_vals.where(test_vals.isin(known), other=le.classes_[0])

        train_df[col] = le.transform(train_vals)
        test_df[col] = le.transform(test_vals)

    if num_cols:
        scaler = StandardScaler()
        train_df[num_cols]=scaler.fit_transform(train_df[num_cols])
        test_df[num_cols]= scaler.transform(test_df[num_cols])

        # Tensors
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["y"].values,         dtype=torch.float32)
    X_test  = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    y_test  = torch.tensor(test_df["y"].values,          dtype=torch.float32)

    return X_train, y_train, X_test, y_test, feature_cols






    

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

    # 
    X_train, y_train, X_test, y_test, feature_cols = build_features(train_df,test_df)

    print("\n=== Returned tensors ===")
    print(f"feature count: {len(feature_cols)}")
    print(f"X_train: shape={tuple(X_train.shape)}, dtype={X_train.dtype}")
    print(f"X_test:  shape={tuple(X_test.shape)}, dtype={X_test.dtype}")
    print(f"y_train: shape={tuple(y_train.shape)}, dtype={y_train.dtype}")
    print(f"y_test:  shape={tuple(y_test.shape)}, dtype={y_test.dtype}")

    print("\n=== Value ranges ===")
    print(f"X_train min: {X_train.min().item():.3f}, max: {X_train.max().item():.3f}, mean: {X_train.mean().item():.3f}")
    print(f"X_test  min: {X_test.min().item():.3f},  max: {X_test.max().item():.3f},  mean: {X_test.mean().item():.3f}")

    print("\n=== Sanity (NaN / Inf) ===")
    print(f"X_train NaN: {torch.isnan(X_train).any().item()}, Inf: {torch.isinf(X_train).any().item()}")
    print(f"X_test  NaN: {torch.isnan(X_test).any().item()},  Inf: {torch.isinf(X_test).any().item()}")

    print("\n=== Labels ===")
    print(f"y_train mean (positive rate): {y_train.mean().item():.3f}")
    print(f"y_test  mean (positive rate): {y_test.mean().item():.3f}")

    print("\n=== First row of X_train (all features) ===")
    for name, val in zip(feature_cols, X_train[0].tolist()):
        print(f"  {name:25s} {val:>10.3f}")

if __name__=="__main__":
    main()