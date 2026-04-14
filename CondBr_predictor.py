import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class BranchDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    

        
    

def load_data(path):

    all_dfs = []
    # Load data from the csv files
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            f_path=os.path.join(path,filename)
            df=pd.read_csv(f_path)
            benchmark_name= filename.replace(".csv","")
            df["benchmark"]=benchmark_name
            all_dfs.append(df)
            print(f"Loaded {filename}")

    df=pd.concat(all_dfs,ignore_index=True) 
    print("Original Shape : ",df.shape)

    #df.iloc[0] Pandas Series
    df["bias_ratio"] = df["Taken"] / df["Executed"] # Bias_ratio column added 
    df["label"] = ((df["bias_ratio"] > 0.98) | (df["bias_ratio"] < 0.02)).astype(int) # Label column added

    col_list=["Opcode", "Routine_Type","Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
        "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5" ]
    
    encoders={}

    for col in col_list:
        le = LabelEncoder()
        df[col]=le.fit_transform(df[col])
        encoders[col]=le

    feauture_cols=[
        "Opcode","Routine_Type","Offset","Size",
        "Prev_Op_1", "Prev_Op_2", "Prev_Op_3", "Prev_Op_4", "Prev_Op_5",
        "Next_Op_1", "Next_Op_2", "Next_Op_3", "Next_Op_4", "Next_Op_5",
        "br_is_loop_header",
        "t_dominates", "t_post_dominates", "t_is_loop_head", "t_is_backedge",
        "f_dominates", "f_post_dominates", "f_is_loop_head", "f_is_backedge",
        "taken_ubd", "fall_ubd", "taken_store", "fall_store"
        ]
    

    print("Biased branches: ",df["label"].sum())
    print("Not Biased: ", (df["label"]==0).sum())


    return df,feauture_cols,encoders



def main():
    Spec_Path="/home/students/cs/2022/ctheod03/Desktop/ADE/results/"
    train_bench=["gcc","povray"]
    test_bench="mcfr"

    df,feature_cols,encoders=load_data(Spec_Path)

    train_df = df[df["benchmark"].isin(train_bench)].copy()
    test_df= df[df["benchmark"]==test_bench].copy()
    if train_df.empty:
        raise ValueError(f"Training dataframe is empty. Check benchmark names: {train_bench}")
    if test_df.empty:
        raise ValueError(f"Test dataframe is empty. Check benchmark name: {test_bench}")
    
    print("\nTrain benchmarks:", train_bench)
    print("Test benchmark  :", test_bench) 

    print("\nTrain size:", len(train_df))
    print("Test size :", len(test_df))

    print("\nTrain class distribution:")
    print("Biased    :", train_df["label"].sum())
    print("Not biased:", (train_df["label"] == 0).sum())

    print("\nTest class distribution:")
    print("Biased    :", test_df["label"].sum())
    print("Not biased:", (test_df["label"] == 0).sum())

    # Extract features/labels
    X_train = train_df[feature_cols].values.astype(np.float32) #Convert to float liek pytorch wants
    y_train = train_df["label"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    # Normalize using only training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets/loaders
    train_dataset = BranchDataset(X_train, y_train)
    test_dataset = BranchDataset(X_test, y_test)

    

if __name__ == "__main__":
    main()