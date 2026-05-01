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
<<<<<<< HEAD
    "t_successor_ends",
    "f_successor_ends",
=======
>>>>>>> a479ebc45ba85762646feebaab6cc866a16a3624
    "Routine_Type",
    "reg1",  "reg1_Op",
    "reg2",  "reg2_Op",
    "reg3",  "reg3_Op",
    "wreg1", "wreg1_Op",
    "wreg2", "wreg2_Op",
    "wreg3", "wreg3_Op",
]

<<<<<<< HEAD
DROP_COLUMNS = ["Taken", "Executed", "Regs_Read",  "Regs_Write"]

OPCODE_COLS = [
    "Flag_Instr_Opcode",
    "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
    "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
    ]

=======
>>>>>>> a479ebc45ba85762646feebaab6cc866a16a3624
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
        
<<<<<<< HEAD
    cols_w = ['wreg1', 'wreg1_Op', 'wreg2', 'wreg2_Op', 'wreg3', 'wreg3_Op']
    cols = ['reg1', 'reg1_Op', 'reg2', 'reg2_Op', 'reg3', 'reg3_Op']

    for df in (train_df,test_df):
=======

    for df in (train_df,test_df):
        cols_w = ['wreg1', 'wreg1_Op', 'wreg2', 'wreg2_Op', 'wreg3', 'wreg3_Op']
        cols = ['reg1', 'reg1_Op', 'reg2', 'reg2_Op', 'reg3', 'reg3_Op']
>>>>>>> a479ebc45ba85762646feebaab6cc866a16a3624
        df[cols] = df["Regs_Read"].apply(get_context, result_type="expand")
        df[cols_w] = df["Regs_Write"].apply(get_context, result_type="expand")


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):

    # Register features splitted (read, write) 
    add_register_features(train_df,test_df)
    
    # Not needed columns
<<<<<<< HEAD
    train_df = train_df.drop(columns = DROP_COLUMNS)
    test_df = test_df.drop(columns = DROP_COLUMNS)

    for col in  OPCODE_COLS:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    

    
=======
    nd_col= {''}
>>>>>>> a479ebc45ba85762646feebaab6cc866a16a3624


    

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

if __name__=="__main__":
    main()