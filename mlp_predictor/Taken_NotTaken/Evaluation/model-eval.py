import os

import numpy as np
import pandas as pd
import torch

from mlp_model import MLP_SS, MLP_ESP, MLP_TNT_emb
from data_engin import load_data, add_register_features, encode_cols, add_label_tnt, get_binary_error
from ml_configs import (
    TEST_FILES, DATA_DIR, DROP_COLUMNS, CT_COLUMNS, ESP_COLUMNS, CONT_COLS, BIN_COLS,
    ROUT_COL, ROUTINE_TYPE_MAP, LANG_COL, LANGUAGE_MAP, REG_COLS, OPC_COLS,
    DataLoader, TensorDataset,
)

MODELS_DIR = "./Taken_NotTaken/Models"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename not in ("SS_CT.pth", "ESP.pth", "EMB_CT.pth"):
            continue

        checkpoint = torch.load(f"{MODELS_DIR}/{filename}",weights_only = False)
        hp = checkpoint["hyperparameters"]

        if filename == "SS_CT.pth":
            model = MLP_SS(
                checkpoint["num_features"],
                1,
                hp["hidden1"],
                hp["hidden2"],
                hp["dropout"]
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        elif filename == "ESP.pth":
            model = MLP_ESP(
                checkpoint["num_features"],
                1,
                hp["hidden1"],
                hp["hidden2"],
                hp["dropout"]
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        
        elif filename == "EMB_CT.pth":
            model = MLP_TNT_emb(
                checkpoint["reg_vocab_size"],
                checkpoint["opc_vocab_size"],
                hp["embed_dim"],
                checkpoint["num_features"],
                1,
                hp["hidden1"],
                hp["hidden2"],
                hp["dropout"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()



if __name__ == "__main__":
    main()
