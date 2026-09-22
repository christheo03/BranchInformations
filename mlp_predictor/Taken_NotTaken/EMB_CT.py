from mlp_model import MLP_TNT_emb
from data_engin import load_data, build_features, add_label_tnt, get_accuracy_error, get_binary_error
from ml_configs import ALL_FILES
from ml_configs import CONT_COLS, BIN_COLS, ROUT_COL, REG_COLS, OPC_COLS
from ml_configs import DataLoader, TensorDataset, torch
from .criterion import WeightedMissLoss
import copy
import optuna
import argparse
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "../../results"
MODEL_NAME = "EMB_CT"
N_CLASSES = 1
EPOCHS = 200


# Prepare the model (Normalization, Model Architecture)
def prepare_datasets(device, train_files, test_files, embed_dim, hidden1, hidden2, dropout, batch_size, min_executed=0):


    # Load raw data from csv files
    train_df = load_data(DATA_DIR, train_files)
    test_df = load_data(DATA_DIR, test_files)

    # Drop training branches with too few executions - their Taken/Executed
    # rate is an unreliable label from too small a sample. Test data is left
    # untouched so evaluation still covers every branch in the held-out file.
    train_df = train_df[train_df["Executed"] >= min_executed].reset_index(drop=True)

    # Add labels
    train_df["y"] = add_label_tnt(train_df)
    test_df["y"] = add_label_tnt(test_df)

    # Snapshot the raw, human-readable rows before build_features drops/
    # encodes columns in place - this is what misclassified branches get
    # exported from later.
    test_df_raw = test_df.copy()

    train_weights = torch.tensor(train_df["weight"].values, dtype=torch.float32)
    train_rates = torch.tensor(train_df["rate"].values, dtype=torch.float32)

    test_weights = torch.tensor(test_df["weight"].values, dtype=torch.float32)
    test_rates = torch.tensor(test_df["rate"].values, dtype=torch.float32)

    # Normalizations
    (
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat, y_test,
        reg_vs, opc_vs,
        scaler, reg_vocab, opc_vocab, numeric_cols,
    ) = build_features(train_df, test_df, return_artifacts=True)

    numeric_features = CONT_COLS + BIN_COLS + [f"{ROUT_COL}_{i}" for i in (1, 2, 3)]
    categorical_features = REG_COLS + OPC_COLS


    g = torch.Generator()


    train_loader = DataLoader(
        TensorDataset(X_train_num, X_train_cat, y_train, train_weights, train_rates),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_num, X_test_cat, y_test, test_weights, test_rates),
        batch_size=batch_size,
        shuffle=False,
    )

    num_features = X_train_num.shape[1]

    model = MLP_TNT_emb(
        reg_vocab_size=reg_vs,
        opc_vocab_size=opc_vs,
        embed_dim=embed_dim,
        num_features=num_features,
        n_classes=N_CLASSES,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    ).to(device)

    return (
        model, train_loader, test_loader, reg_vs, opc_vs, num_features,
        scaler, reg_vocab, opc_vocab, numeric_cols, test_df_raw,
    )


def train(model, train_loader, test_loader, loss_func, optim, device, patience=25, trial=None):
    best_test_err = float('inf')
    patience_counter = 0
    best_model_state = None
    saved_epoch_test_err = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_train_err, total_train_n = 0.0, 0.0, 0

        for x_num, x_cat, y_target, weights, rates in train_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            weights, rates = weights.to(device), rates.to(device)
            y_target = y_target.to(device).float()

            optim.zero_grad()

            outputs = model(x_num, x_cat)
            loss = loss_func(outputs, y_target, weights, rates)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * x_num.shape[0]
            epoch_train_err += get_binary_error(outputs, weights,rates)
            total_train_n += x_num.shape[0]

        epoch_loss /= total_train_n
        epoch_train_err /= total_train_n

        model.eval()
        epoch_test_err, total_test_n = 0.0, 0
        with torch.no_grad():
            for x_num, x_cat, y_target, weights, rates in test_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                weights, rates = weights.to(device), rates.to(device)
                y_target = y_target.to(device).float()

                outputs = model(x_num, x_cat)
                epoch_test_err += get_binary_error(outputs, weights,rates)
                total_test_n += x_num.shape[0]
        epoch_test_err /= total_test_n


        # Optuna pruning check (stops runs early that clearly won't win)
        if trial is not None:
            trial.report(epoch_test_err, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if epoch_test_err < best_test_err:
            best_test_err = epoch_test_err
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            saved_epoch_test_err = epoch_test_err
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return saved_epoch_test_err


def objective(trial, train_files, test_files, min_executed=0):
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suggest parameters for this trial
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    embed_dim = trial.suggest_categorical("embed_dim", [4, 8, 16, 24])
    hidden1 = trial.suggest_int("hidden1", 64,512, step=64)
    hidden2 = trial.suggest_int("hidden2", 32,256,step=32)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])

    # Load dynamic datasets and instantiate the model
    model, train_loader, test_loader, reg_vs, opc_vs, num_features, scaler, reg_vocab, opc_vocab, numeric_cols, _ = prepare_datasets(
        device, train_files, test_files,
        embed_dim=embed_dim, hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        batch_size=batch_size, min_executed=min_executed
    )

    loss_function = WeightedMissLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    try:
        miss_rate = train(model, train_loader, test_loader, loss_function, optimizer, device, trial=trial)
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
    return miss_rate


# Runs the held-out fold's test set through the trained model once and
# reports 1 - accuracy - no checkpoint is saved, this is purely for the
# leave-one-out evaluation numbers.
def evaluate_fold(model, test_loader, device):
    model.eval()
    all_outputs, all_y, all_weights, all_rates = [], [], [], []

    with torch.no_grad():
        for x_num, x_cat, y_target, weights, rates in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            outputs = model(x_num, x_cat)

            all_outputs.append(outputs.cpu())
            all_y.append(y_target)
            all_weights.append(weights)
            all_rates.append(rates)

    outputs = torch.cat(all_outputs)
    y = torch.cat(all_y)
    weights = torch.cat(all_weights)
    rates = torch.cat(all_rates)

    preds = (outputs.squeeze(-1) > 0.5).float()
    one_minus_accuracy = (preds != y).float().mean().item()

    # Weighted miss-rate metric - not what AccuracyLoss trains toward anymore,
    # reported here purely as a diagnostic alongside 1-accuracy.
    binary_error = get_binary_error(outputs, weights, rates) / weights.sum().item()

    return one_minus_accuracy, binary_error, preds, y


# Writes every branch in the test file (raw features + predicted/actual
# label) from the final, fully-trained model to one CSV - correct and
# incorrect predictions together.
def export_predictions(test_df_raw, preds, y_true, test_file):
    predictions = test_df_raw.copy()
    predictions["predicted"] = preds.numpy().astype(int)
    predictions["actual"] = y_true.numpy().astype(int)

    out_dir = "./predictions"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_NAME}_{test_file}.csv")
    predictions.to_csv(out_path, index=False)
    return out_path, len(predictions)


# Runs one LOO fold end-to-end (search + retrain + eval) for a single
# held-out file, so this can be dispatched to its own machine/process.
def run_fold(test_file, device, min_executed=0):
    train_files = [f for f in ALL_FILES if f != test_file]
    test_files = [test_file]


    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_files, test_files, min_executed=min_executed), n_trials=80)

    best_params = study.best_params

    model, train_loader, test_loader, reg_vs, opc_vs, num_features, scaler, reg_vocab, opc_vocab, numeric_cols, test_df_raw = prepare_datasets(
        device, train_files, test_files,
        embed_dim=best_params["embed_dim"],
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"],
        min_executed=min_executed,
    )

    loss_function = WeightedMissLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    # Train one final time to converge on the optimal weights - not saved.
    train(model, train_loader, test_loader, loss_function, optimizer, device)

    one_minus_accuracy, binary_error, preds, y_true = evaluate_fold(model, test_loader, device)

    predictions_path, predictions_count = export_predictions(test_df_raw, preds, y_true, test_file)

    return {
        "test_file": test_file,
        "one_minus_accuracy": one_minus_accuracy,
        "binary_error": binary_error,
        "predictions_path": predictions_path,
        "predictions_count": predictions_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file", required=True, choices=ALL_FILES,
        help="Run this fold (held out as test).",
    )
    parser.add_argument(
        "--min-executed", type=int, default=0,
        help="Drop training branches with Executed below this count (test set is untouched).",
    )
    args = parser.parse_args()

    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = run_fold(args.test_file, device, min_executed=args.min_executed)

    print(f"\n\n{'#' * 50}\n{args.test_file}\n{'#' * 50}")
    print(f"  {result['test_file']:<20} Static - Loss: {result['one_minus_accuracy'] * 100:7.3f}%   Dynamic Weighted loss: {result['binary_error'] * 100:7.3f}%")
    print(f"  Predictions: {result['predictions_count']} branches -> {result['predictions_path']}")


if __name__ == "__main__":
    main()
