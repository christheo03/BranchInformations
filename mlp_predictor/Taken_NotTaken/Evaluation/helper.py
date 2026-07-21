import numpy as np
import pandas as pd
import torch

from mlp_model import MLP_SS, MLP_ESP, MLP_TNT_emb
from data_engin import encode_cols, get_binary_error
from ml_configs import (
    CONT_COLS, BIN_COLS,
    ROUT_COL, ROUTINE_TYPE_MAP, LANG_COL, LANGUAGE_MAP, REG_COLS, OPC_COLS,
)

def normalize_ss(df, checkpoint, columns):
    """
    normalizes the dataframe's data for the model

    ARGS:
        df: The dataframe holding the data
        checkpoint: Saved parameters of the model (vocabs, scaler)
        columns: Features that need normalization

    Returns:
        The normalized scaled data 
    """
    for col in [c for c in CONT_COLS + BIN_COLS if c in columns]:
        df[col] = df[col].fillna(0)

    if ROUT_COL in columns:
        df[ROUT_COL] = df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0)
    if LANG_COL in columns:
        df[LANG_COL] = df[LANG_COL].map(LANGUAGE_MAP).fillna(0)

    df = df[columns].copy()

    reg_cols = [c for c in REG_COLS if c in columns]
    opc_cols = [c for c in OPC_COLS if c in columns]
    for col in reg_cols + opc_cols:
        df[col] = df[col].fillna("-1").astype(str)

    for group, vocab_key in ((reg_cols, "reg_vocab"), (opc_cols, "opc_vocab")):
        if group:
            encoded = encode_cols(df, group, checkpoint[vocab_key])
            for i, col in enumerate(group):
                df[col] = encoded[:, i]

    return torch.tensor(checkpoint["scaler"].transform(df), dtype=torch.float32)


def build_ss_ct(checkpoint):
    hp = checkpoint["hyperparameters"]
    model = MLP_SS(checkpoint["num_features"], 1, hp["hidden1"], hp["hidden2"], hp["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_esp(checkpoint):
    hp = checkpoint["hyperparameters"]
    model = MLP_ESP(checkpoint["num_features"], 1, hp["hidden1"], hp["hidden2"], hp["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_emb_ct(checkpoint):
    hp = checkpoint["hyperparameters"]
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
    return model






# EMB_CT keeps numeric/categorical separate: numeric cols + one-hot
# Routine_Type get scaled into X_num, registers/opcodes get embedding-ready
# ids in X_cat.
def normalize_emb(df, checkpoint,columns):
    df = df[columns].copy()

    for col in CONT_COLS + BIN_COLS:
        df[col] = df[col].fillna(0)
    df[ROUT_COL] = df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0).astype(int)

    # Same 3-category one-hot (no dummy for "missing") as data_engin.build_features.
    X_rout = pd.get_dummies(
        pd.Categorical(df[ROUT_COL], categories=[1, 2, 3]),
        prefix=ROUT_COL,
    ).astype(float).values

    X_scaled = checkpoint["scaler"].transform(df[checkpoint["numeric_cols"]])
    X_num = torch.tensor(np.hstack([X_scaled, X_rout]), dtype=torch.float32)

    for col in REG_COLS + OPC_COLS:
        df[col] = df[col].fillna("-1").astype(str)

    X_cat = torch.tensor(
        np.hstack([
            encode_cols(df, REG_COLS, checkpoint["reg_vocab"]),
            encode_cols(df, OPC_COLS, checkpoint["opc_vocab"]),
        ]),
        dtype=torch.long,
    )

    return X_num, X_cat


def evaluate(outputs, y, weights, rates, executed):
    """
    Turns raw model outputs into weighted miss rate, dynamic miss rate,
    static misprediction rate, and a confusion table.

    ARGS:
        outputs: Model predictions, one "P(Taken)" value per row in [0, 1]
        y: True Taken/Not-Taken label for each row
        weights: Per-benchmark-normalized execution weight for each row
                 (sums to 1 within each benchmark)
        rates: True taken-rate for each row (used by get_binary_error)
        executed: Raw execution count for each row (the actual number of
                  times that branch ran, not normalized)

    Returns:
        dict with keys: weighted_miss_rate, dynamic_miss_rate,
        static_miss_rate, confusion
    """
    # Weighted (macro-average across benchmarks): since each benchmark's
    # `weights` column sums to 1, this equals the average of the individual
    # per-benchmark miss rates - every benchmark counts equally regardless
    # of how many instructions it actually ran.
    weighted_miss_rate = get_binary_error(outputs, weights, rates) / torch.sum(weights).item()

    # Dynamic (micro-average): all mispredicted executions across every
    # branch in the whole test set, divided by all executions - uses the
    # raw Executed counts, so a benchmark that ran more instructions
    # proportionally has more say in this number.
    dynamic_miss_rate = get_binary_error(outputs, executed, rates) / torch.sum(executed).item()

    preds = (outputs.squeeze(-1) > 0.5).float()

    # Static: one vote per branch regardless of execution count - plain
    # wrong-hard-predictions / total-predictions (same as 1 - accuracy).
    static_miss_rate = (preds != y).float().mean().item()

    confusion = {
        "TP": ((preds == 1) & (y == 1)).sum().item(),  # predicted taken, actually taken
        "TN": ((preds == 0) & (y == 0)).sum().item(),  # predicted not-taken, actually not-taken
        "FP": ((preds == 1) & (y == 0)).sum().item(),  # predicted taken, actually not-taken
        "FN": ((preds == 0) & (y == 1)).sum().item(),  # predicted not-taken, actually taken
    }

    return {
        "weighted_miss_rate": weighted_miss_rate,
        "dynamic_miss_rate": dynamic_miss_rate,
        "static_miss_rate": static_miss_rate,
        "confusion": confusion,
    }


def mispredicted_branches(outputs, y, test_df):
    """
    Rows where the model's hard prediction disagrees with the true label,
    sorted by Executed (descending) - the branches that would cause the
    most real mispredictions show up first.

    ARGS:
        outputs: Model predictions, one "P(Taken)" value per row in [0, 1]
        y: True Taken/Not-Taken label for each row
        test_df: The raw test dataframe (same row order as outputs/y)

    Returns:
        pd.DataFrame of the mispredicted rows only
    """
    preds = (outputs.squeeze(-1) > 0.5).float()
    mask = (preds != y).numpy().astype(bool)

    mispred = test_df[mask].copy()
    mispred["Predicted"] = preds[mask].numpy().astype(int)

    return mispred.sort_values("Executed", ascending=False)