from mlp_model import MLP_SS
from data_engin import load_data, add_register_features, std_scaler_norm, add_label_tnt, get_binary_error
from ml_configs import ALL_FILES, CT_COLUMNS
from ml_configs import DataLoader, TensorDataset, torch
from .criterion import WeightedMissLoss
import copy
import optuna
import argparse

DATA_DIR = "../../results"
N_CLASSES = 1
EPOCHS = 200

# Prepare the model (Normalization, Model Architecture)
def prepare_datasets(device, train_files, test_files, hidden1, hidden2, dropout, batch_size):


    # Load raw data from csv files
    train_df = load_data(DATA_DIR, train_files)
    test_df = load_data(DATA_DIR, test_files)

    # Add labels
    train_df["y"] = add_label_tnt(train_df)
    test_df["y"] = add_label_tnt(test_df)

    train_weights = torch.tensor(train_df["weight"].values, dtype=torch.float32)
    train_rates = torch.tensor(train_df["rate"].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["y"].values, dtype=torch.float32)

    test_weights = torch.tensor(test_df["weight"].values, dtype=torch.float32)
    test_rates = torch.tensor(test_df["rate"].values, dtype=torch.float32)
    y_test = torch.tensor(test_df["y"].values, dtype=torch.float32)

    add_register_features(train_df, test_df)

    X_train, X_test, scaler, reg_vocab, opc_vocab, feature_columns = std_scaler_norm(
        train_df, test_df, CT_COLUMNS
    )

    print(f"[SS_CT] Features used ({len(feature_columns)}): {feature_columns}")
    print(f"X_train shape: {X_train.shape}")

    # Initialize DataLoaders
    g = torch.Generator()


    train_loader = DataLoader(
        TensorDataset(X_train, y_train, train_weights, train_rates),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )


    test_loader = DataLoader(
        TensorDataset(X_test, y_test, test_weights, test_rates),
        batch_size=batch_size,
        shuffle=False,
    )

    num_features = X_train.shape[1]

    model = MLP_SS(
        num_features=num_features,
        n_classes=N_CLASSES,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    ).to(device)

    return model, train_loader, test_loader, num_features, scaler, reg_vocab, opc_vocab, feature_columns



def train(model, train_loader, test_loader, loss_func, optim, device, patience=25, trial=None):
    best_test_err = float('inf')
    patience_counter = 0
    best_model_state = None
    saved_epoch_test_err = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_train_err, total_train_w = 0.0, 0.0, 0.0

        for x, y_target, weights, rates in train_loader:
            x,weights, rates =x.to(device), weights.to(device), rates.to(device)
            y_target = y_target.to(device).float()

            optim.zero_grad()

            outputs = model(x)
            loss = loss_func(outputs, y_target, weights, rates)

            loss.backward()
            optim.step()

            epoch_loss += loss.item() * torch.sum(weights).item()
            epoch_train_err += get_binary_error(outputs, weights, rates)
            total_train_w += torch.sum(weights).item()

        epoch_loss /= total_train_w
        epoch_train_err /= total_train_w

        model.eval()
        epoch_test_err, total_test_w = 0.0, 0.0
        with torch.no_grad():
            for x, _, weights, rates in test_loader:
                x,weights, rates = x.to(device),weights.to(device), rates.to(device)


                outputs = model(x)
                epoch_test_err += get_binary_error(outputs, weights, rates)
                total_test_w += torch.sum(weights).item()
        epoch_test_err /= total_test_w

        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.5f} | Train Miss %: {epoch_train_err * 100:.3f}% | Test Miss %: {epoch_test_err * 100:.3f}%")

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
                print(f"\n[EARLY STOPPING] No improvement in test miss rate for {patience} epochs. Stopping.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return saved_epoch_test_err


def objective(trial, train_files, test_files):
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suggest parameters for this trial
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    hidden1 = trial.suggest_int("hidden1", 64,512, step=64)
    hidden2 = trial.suggest_int("hidden2", 32,256,step=32)
    batch_size = trial.suggest_categorical("batch_size", [128,256, 512, 1024])


    model, train_loader, test_loader, num_features, scaler, reg_vocab, opc_vocab, feature_columns = prepare_datasets(
        device, train_files, test_files,
        hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        batch_size=batch_size
    )

    loss_function = WeightedMissLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    try:
        miss_rate = train(model, train_loader, test_loader, loss_function, optimizer, device, trial=trial)
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
    return miss_rate


# Runs the held-out fold's test set through the trained model once and
# reports (1 - accuracy) and the raw get_binary_error value - no checkpoint
# is saved, this is purely for the leave-one-out evaluation numbers.
def evaluate_fold(model, test_loader, device):
    model.eval()
    all_outputs, all_y, all_weights, all_rates = [], [], [], []

    with torch.no_grad():
        for x, y_target, weights, rates in test_loader:
            x, weights, rates = x.to(device), weights.to(device), rates.to(device)
            outputs = model(x)

            all_outputs.append(outputs.cpu())
            all_y.append(y_target)
            all_weights.append(weights.cpu())
            all_rates.append(rates.cpu())

    outputs = torch.cat(all_outputs)
    y = torch.cat(all_y)
    weights = torch.cat(all_weights)
    rates = torch.cat(all_rates)

    preds = (outputs.squeeze(-1) > 0.5).float()
    one_minus_accuracy = (preds != y).float().mean().item()
    binary_error = get_binary_error(outputs, weights, rates)

    return one_minus_accuracy, binary_error


# Runs one LOO fold end-to-end (search + retrain + eval) for a single
# held-out file, so this can be dispatched to its own machine/process.
def run_fold(test_file, device):
    train_files = [f for f in ALL_FILES if f != test_file]
    test_files = [test_file]

    print(f"\n\n{'#' * 50}\nLEAVE-ONE-OUT FOLD - held out: {test_file}\n{'#' * 50}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_files, test_files), n_trials=75)

    best_params = study.best_params
    print(f"[{test_file}] Best Hyperparameters: {best_params}")
    print(f"[{test_file}] Best Test Miss Rate during search: {study.best_value * 100:.3f}%")

    model, train_loader, test_loader, num_features, scaler, reg_vocab, opc_vocab, feature_columns = prepare_datasets(
        device, train_files, test_files,
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"]
    )

    loss_function = WeightedMissLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    # Train one final time to converge on the optimal weights - not saved.
    train(model, train_loader, test_loader, loss_function, optimizer, device)

    one_minus_accuracy, binary_error = evaluate_fold(model, test_loader, device)

    print(f"[{test_file}] 1 - Accuracy:     {one_minus_accuracy * 100:.3f}%")
    print(f"[{test_file}] get_binary_error: {binary_error:.6f}")

    return {"test_file": test_file, "one_minus_accuracy": one_minus_accuracy, "binary_error": binary_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-file", required=True, choices=ALL_FILES,
        help="Run this fold (held out as test).",
    )
    args = parser.parse_args()

    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = run_fold(args.test_file, device)

    print(f"\n\n{'#' * 50}\nRESULT\n{'#' * 50}")
    print(f"  {result['test_file']:<20} 1-Accuracy: {result['one_minus_accuracy'] * 100:7.3f}%   get_binary_error: {result['binary_error']:.6f}")


if __name__ == "__main__":
    main()
