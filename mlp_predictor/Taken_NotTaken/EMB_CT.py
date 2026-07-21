from mlp_model import MLP_TNT_emb
from data_engin import load_data, build_features, add_label_tnt, get_binary_error
from ml_configs import TEST_FILES, TRAIN_FILES
from ml_configs import CONT_COLS, BIN_COLS, ROUT_COL, REG_COLS, OPC_COLS
from ml_configs import DataLoader, TensorDataset, torch
from .criterion import DynamicMissPredictionLoss
import copy
import optuna

DATA_DIR = "../../results"
N_CLASSES = 1
EPOCHS = 200


# Prepare the model (Normalization, Model Architecture)
# RETURN VALUE CHANGED: Now returns reg_vs, opc_vs, and num_features to save them at the end.
def prepare_datasets(device, train_files, test_files, embed_dim, hidden1, hidden2, dropout, batch_size):
    

    # Load raw data from csv files
    train_df = load_data(DATA_DIR, train_files)
    test_df = load_data(DATA_DIR, test_files)

    # Add labels
    train_df["y"] = add_label_tnt(train_df)
    test_df["y"] = add_label_tnt(test_df)

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
    print(f"[EMB_CT] Scaled numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"[EMB_CT] Embedded categorical features ({len(categorical_features)}): {categorical_features}")

    print(f"[EMB_CT] Categorial data: {X_train_cat.shape}")
    print(f"[EMB_CT] Numerical data: {X_train_num.shape}")

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
        scaler, reg_vocab, opc_vocab, numeric_cols,
    )


def train(model, train_loader, test_loader, loss_func, optim, device, patience=25, trial=None):
    best_test_err = float('inf')  
    patience_counter = 0
    best_model_state = None
    saved_epoch_test_err = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_train_err, total_train_w = 0.0, 0.0, 0.0
        
        for x_num, x_cat, y_target, weights, rates in train_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            weights, rates = weights.to(device), rates.to(device)
            y_target = y_target.to(device).float()

            optim.zero_grad()

            outputs = model(x_num, x_cat)
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
            for x_num, x_cat, _, weights, rates in test_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                weights, rates = weights.to(device), rates.to(device)
                
                outputs = model(x_num, x_cat)
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


def objective(trial):
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Suggest parameters for this trial
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    embed_dim = trial.suggest_categorical("embed_dim", [4, 8, 16,24])
    hidden1 = trial.suggest_int("hidden1", 64,512, step=64)
    hidden2 = trial.suggest_int("hidden2", 32,256,step=32)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    
    # Load dynamic datasets and instantiate the model
    model, train_loader, test_loader, reg_vs, opc_vs, num_features, scaler, reg_vocab, opc_vocab, numeric_cols = prepare_datasets(
        device, TRAIN_FILES, TEST_FILES,
        embed_dim=embed_dim, hidden1=hidden1, hidden2=hidden2, dropout=dropout,
        batch_size=batch_size
    )
    
    loss_function = DynamicMissPredictionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    try:
        miss_rate = train(model, train_loader, test_loader, loss_function, optimizer, device, trial=trial)
    except optuna.exceptions.TrialPruned:
        raise optuna.exceptions.TrialPruned()
    return miss_rate


def main():
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Starting automated Hyperparameter optimization...")
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    # Run search for 20 trials (training runs)
    study.optimize(objective, n_trials=75)
    
    best_params = study.best_params

    # Print results
    print(f"\n\n{'#' * 50}\nOPTIMIZATION COMPLETE\n{'#' * 50}")
    print("Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key:<15} : {value}")
    print(f"\nBest Test Miss Rate: {study.best_value * 100:.3f}%")
    print(f"{'#' * 50}\n")

    # Train the final model using the best hyperparameters found
    print("Training final model using best hyperparameters...")
    model, train_loader, test_loader, reg_vs, opc_vs, num_features, scaler, reg_vocab, opc_vocab, numeric_cols = prepare_datasets(
        device, TRAIN_FILES, TEST_FILES,
        embed_dim=best_params["embed_dim"],
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"]
    )

    loss_function = DynamicMissPredictionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    # Train one final time to converge on the optimal weights
    train(model, train_loader, test_loader, loss_function, optimizer, device)

    # Save the model weights together with everything needed to preprocess
    # raw test CSVs the same way at inference time (scaler + vocabs), so the
    # checkpoint is self-sufficient and doesn't require re-fitting on
    # TRAIN_FILES to evaluate later.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": best_params,
        "reg_vocab_size": reg_vs,
        "opc_vocab_size": opc_vs,
        "num_features": num_features,
        "scaler": scaler,
        "reg_vocab": reg_vocab,
        "opc_vocab": opc_vocab,
        "numeric_cols": numeric_cols,
    }

    torch.save(checkpoint, "EMB_CT.pth")
    print("Successfully saved best model and all parameters to 'EMB_CT.pth'!")

    
if  __name__ == "__main__":
    main()