from mlp_model import set_seed, MLP_ESP
from data_engin import load_data, add_register_features,pd
from ml_configs import SEED, TEST_FILES, TRAIN_FILES, DROP_COLUMNS, CONT_COLS, BIN_COLS, ROUT_COL,ROUTINE_TYPE_MAP
from ml_configs import DataLoader, TensorDataset, torch,StandardScaler
from .criterion import DynamicMissPredictionLoss
import copy
import optuna

DATA_DIR = "../../results"
N_CLASSES = 1
EPOCHS = 500


# Add taken, not taken label
def add_label(df):
    rate = df["Taken"] / df["Executed"]
    label = (rate > 0.5).astype(int)
    return label


# Prepare the model (Normalization, Model Architecture)
def prepare_datasets(device, train_files, test_files, hidden1, hidden2, dropout, batch_size):
    set_seed(SEED)

    # Load raw data from csv files
    train_df = load_data(DATA_DIR, train_files)
    test_df = load_data(DATA_DIR, test_files)

    # Add labels
    train_df["y"] = add_label(train_df)
    test_df["y"] = add_label(test_df)

    train_weights = torch.tensor(train_df["weight"].values, dtype=torch.float32)
    train_rates = torch.tensor(train_df["rate"].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["y"].values, dtype=torch.float32)

    test_weights = torch.tensor(test_df["weight"].values, dtype=torch.float32)
    test_rates = torch.tensor(test_df["rate"].values, dtype=torch.float32)
    y_test = torch.tensor(test_df["y"].values, dtype=torch.float32)

    add_register_features(train_df, test_df)
    
    scaler = StandardScaler()

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
    
    train_df[ROUT_COL] = train_df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0)
    test_df[ROUT_COL] = test_df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0)

    string_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if ROUT_COL in string_cols:
        string_cols.remove(ROUT_COL)

    for col in string_cols:
        unique_vals = pd.concat([train_df[col], test_df[col]]).dropna().unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        train_df[col] = train_df[col].map(mapping).fillna(0)
        test_df[col] = test_df[col].map(mapping).fillna(0)

    
    X_train_scaled = scaler.fit_transform(train_df)
    X_test_scaled = scaler.transform(test_df)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Initialize DataLoaders
    g = torch.Generator()
    g.manual_seed(SEED)

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

    model = MLP_ESP(
        num_features=num_features,
        n_classes=N_CLASSES,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    ).to(device)


    
    return model, train_loader, test_loader, num_features

def get_binary_error(outputs, weights, rates):
    y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
    y_k_binary = (y_k > 0.5).float()
    errors = ((1.0 - y_k_binary) * rates * weights) + (y_k_binary * (1.0 - rates) * weights)
    return torch.sum(errors).item()


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


def objective(trial):
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # Suggest parameters for this trial
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden1 = trial.suggest_int("hidden1", 64,512, step=64)
    hidden2 = trial.suggest_int("hidden2", 32,256,step=32)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    

    model, train_loader, test_loader, num_features = prepare_datasets(
        device, TRAIN_FILES, TEST_FILES, 
        hidden1=hidden1, hidden2=hidden2, dropout=dropout,
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
    study.optimize(objective, n_trials=20)
    
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
    model, train_loader, test_loader, num_features = prepare_datasets(
        device, TRAIN_FILES, TEST_FILES, 
        hidden1=best_params["hidden1"],
        hidden2=best_params["hidden2"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"]
    )
    
    loss_function = DynamicMissPredictionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    
    # Train one final time to converge on the optimal weights
    train(model, train_loader, test_loader, loss_function, optimizer, device)

    # Save the model state dictionary and all config parameters to rebuild it later
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": best_params,
        "num_features": num_features
    }
    
    torch.save(checkpoint, "best_esp_model.pth")
    print("Successfully saved best model and all parameters to 'best_model.pth'!")

    
if  __name__ == "__main__":
    main()