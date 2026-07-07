from ml_configs import os,torch,pd,np,re
from ml_configs import (CLASS_HNT,CLASS_NB,CLASS_HT,
                        DROP_COLUMNS,BIN_COLS,CONT_COLS,ROUT_COL,REG_COLS,OPC_COLS,
                        ROUTINE_TYPE_MAP)
from ml_configs import StandardScaler

# Load raw data from csv files  
def load_data(data_dir, names):
    dfs = []

    for filename in names:
        path = os.path.join(data_dir, filename + ".csv")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing csv: {path}")

        dfs.append(pd.read_csv(path))

    return pd.concat(dfs, ignore_index=True)


# Add label on each row  (HT,HNT,NB) baised on threshold
def add_label(df, thr=0.005):
    rate = df["Taken"] / df["Executed"]

    label = pd.Series(CLASS_NB, index=df.index)
    label[rate < thr] = CLASS_HNT
    label[rate > 1 - thr] = CLASS_HT

    return label.astype(int)

# Register column that stores all the values is seperated to features
def add_register_features(train_df, test_df):
    def get_read_context(text):
        if pd.isna(text) or str(text).strip() == "":
            return "-1", "-1", "-1", "-1", "-1", "-1"

        matches = re.findall(
            r"([a-zA-Z0-9_]+)\(([^)]+)\)",
            str(text)
        )

        result = []

        for name, value in matches:
            result.append(name)
            result.append(value)

        while len(result) < 6:
            result.extend(["-1", "-1"])

        return result[0], result[1], result[2], result[3], result[4], result[5]

    def get_write_context(text):
        if pd.isna(text) or str(text).strip() == "":
            return "-1", "-1"

        parts = [p.strip() for p in str(text).split() if p.strip()]

        while len(parts) < 2:
            parts.append("-1")

        return parts[0], parts[1]

    read_cols = [
        "reg1", "reg1_Op",
        "reg2", "reg2_Op",
        "reg3", "reg3_Op",
    ]

    write_cols = [
        "wreg1",
        "wreg2",
    ]

    for df in (train_df, test_df):
        df[read_cols] = pd.DataFrame(
            df["Regs_Read"].apply(get_read_context).tolist(),
            index=df.index,
        )

        df[write_cols] = pd.DataFrame(
            df["Regs_Write"].apply(get_write_context).tolist(),
            index=df.index,
        )


# Build a map that matches strings to integers
# Cols are the columns that are related 
# Ex. all opcode columns
def build_shared_vocab(train_df, cols):
    all_vals = set()

    for col in cols:
        all_vals.update(train_df[col].astype(str).unique())

    return {
        val: idx + 1
        for idx, val in enumerate(sorted(all_vals))
    }


def encode_cols(df, cols, vocab):
    return np.stack(
        [
            df[col].astype(str).map(lambda x: vocab.get(x, 0)).values
            for col in cols
        ],
        axis=1,
    )

# Normalizations
def build_features(train_df, test_df):
    # Add features about registes read and reagister defined opcodes
    add_register_features(train_df, test_df)

    # Drop columns not needed 
    train_df.drop(
        columns=[c for c in DROP_COLUMNS if c in train_df.columns], inplace= True 
    )

    test_df.drop(
        columns=[c for c in DROP_COLUMNS if c in test_df.columns], inplace = True
    )

    # fill null 
    for col in CONT_COLS + BIN_COLS:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)

    # Routine_Type is a small fixed categorical feature:
    # NonLeaf / Leaf / Recursive.
    train_df[ROUT_COL] = (
        train_df[ROUT_COL]
        .map(ROUTINE_TYPE_MAP)
        .fillna(0)
        .astype(int)
    )

    test_df[ROUT_COL] = (
        test_df[ROUT_COL]
        .map(ROUTINE_TYPE_MAP)
        .fillna(0)
        .astype(int)
    )

    routine_categories = [0, 1, 2, 3]

    X_train_rout = pd.get_dummies(
        pd.Categorical(train_df[ROUT_COL], categories=routine_categories),
        prefix=ROUT_COL,
    ).astype(float).values

    X_test_rout = pd.get_dummies(
        pd.Categorical(test_df[ROUT_COL], categories=routine_categories),
        prefix=ROUT_COL,
    ).astype(float).values


    # Normilize numeric values
    NUMERIC_COLS = CONT_COLS + BIN_COLS

    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLS])

    X_train_scaled = scaler.transform(train_df[NUMERIC_COLS])
    X_test_scaled = scaler.transform(test_df[NUMERIC_COLS])


    X_train_num = torch.tensor(
        np.hstack([X_train_scaled, X_train_rout]),
        dtype=torch.float32,
    )

    X_test_num = torch.tensor(
        np.hstack([X_test_scaled, X_test_rout]),
        dtype=torch.float32,
    )


    # Normilize Categorial columns
    for col in REG_COLS + OPC_COLS:
        train_df[col] = train_df[col].fillna("-1").astype(str)
        test_df[col] = test_df[col].fillna("-1").astype(str)

    # Map regs to ID
    reg_vocab = build_shared_vocab(train_df, REG_COLS)
    opc_vocab = build_shared_vocab(train_df, OPC_COLS)

    X_train_cat = torch.tensor(
        np.hstack([
            encode_cols(train_df, REG_COLS, reg_vocab),
            encode_cols(train_df, OPC_COLS, opc_vocab),
        ]),
        dtype=torch.long,
    )




    X_test_cat = torch.tensor(
        np.hstack([
            encode_cols(test_df, REG_COLS, reg_vocab),
            encode_cols(test_df, OPC_COLS, opc_vocab),
        ]),
        dtype=torch.long,
    )

    y_train = torch.tensor(train_df["y"].values, dtype=torch.long)
    y_test = torch.tensor(test_df["y"].values, dtype=torch.long)

    return (
        X_train_num,
        X_train_cat,
        y_train,
        X_test_num,
        X_test_cat,
        y_test,
        len(reg_vocab) + 1,
        len(opc_vocab) + 1,
    )