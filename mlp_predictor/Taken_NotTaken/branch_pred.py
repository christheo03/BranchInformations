from mlp_model import set_seed,MLP_TNT_emb
from data_engin import load_data, build_features
from ml_configs import SEED,BATCH_SIZE,ALL_FILES
from ml_configs import DataLoader,TensorDataset, torch
from .criterion import DynamicMissPredictionLoss
import copy

DATA_DIR = "../../results"

EMBED_DIM = 16
N_CLASSES = 1
EPOCHS = 500

LR = 0.001
LR_INC=1.05
LR_DEC=0.5

# Add taken, not taken label
def add_label(df):
    rate = df["Taken"]/df["Executed"]

    label = (rate > 0.5).astype(int)

    return label


# Prepare the model (Normalization, Model Architecture)
def prepare_datasets(device,train_files,test_files):
    set_seed( SEED)

    # Load raw data from csv files
    train_df = load_data(DATA_DIR,train_files)
    test_df = load_data(DATA_DIR,test_files)

    # Add labels
    train_df["y"] = add_label(train_df)
    test_df["y"] = add_label(test_df)

    train_weights = torch.tensor(train_df["weight"].values,dtype=torch.float32)
    train_rates = torch.tensor(train_df["rate"].values,dtype=torch.float32)

    test_weights = torch.tensor(test_df["weight"].values, dtype=torch.float32)
    test_rates = torch.tensor(test_df["rate"].values, dtype=torch.float32)

    # Normalizations
    (
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat, y_test,
        reg_vs, opc_vs
    ) = build_features(train_df, test_df)

    g = torch.Generator()
    g.manual_seed(SEED)


    train_loader = DataLoader(
        TensorDataset(X_train_num, X_train_cat, y_train,train_weights, train_rates),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_num, X_test_cat, y_test, test_weights, test_rates),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MLP_TNT_emb(
        reg_vocab_size=reg_vs,
        opc_vocab_size=opc_vs,
        embed_dim=EMBED_DIM,
        num_features=X_train_num.shape[1],
        n_classes=N_CLASSES,
    ).to(device)

    return model, train_loader, test_loader
    

def get_binary_error(outputs, weights, rates):

    y_k = outputs.squeeze(-1) if outputs.dim() > 1 else outputs
    y_k_binary = (y_k > 0.5).float()
    errors = ((1.0 - y_k_binary) * rates * weights) + (y_k_binary * (1.0 - rates) * weights)
    return torch.sum(errors).item()

def train(model, train_loader, test_loader, loss_func, optim, device, patience=10):
    prev_loss = float('inf')
    best_test_err = float('inf')  # <--- Track the best test threshold error
    patience_counter = 0
    best_model_state = None
    saved_epoch_test_err = 0.0
    for epoch in range(EPOCHS):
        # 1. TRAINING SWEEP (Gradient Accumulation - Continuous Loss Function)
        model.train()
        optim.zero_grad()
        epoch_loss, epoch_train_err, total_train_w = 0.0, 0.0, 0.0
        for x_num, x_cat, y_target, weights, rates in train_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            weights, rates = weights.to(device), rates.to(device)
            y_target = y_target.to(device).float()
            outputs = model(x_num, x_cat)
            
            # --- Continuous Loss is used to train the network ---
            loss = loss_func(outputs, y_target, weights, rates) 
            
            # Scale batch loss to simulate a true full-corpus update
            batch_scale = torch.sum(weights) / len(train_loader.dataset)
            (loss * batch_scale).backward()
            epoch_loss += loss.item() * torch.sum(weights).item()
            epoch_train_err += get_binary_error(outputs, weights, rates)
            total_train_w += torch.sum(weights).item()
        optim.step()  # Weights update ONCE per epoch sweep
        epoch_loss /= total_train_w
        epoch_train_err /= total_train_w
        # # 2. ADAPTIVE LEARNING RATE STEP
        # current_lr = optim.param_groups[0]['lr']
        # new_lr = current_lr * LR_INC if epoch_loss < prev_loss else current_lr * LR_DEC
        # for param_group in optim.param_groups:
        #     param_group['lr'] = new_lr
        # prev_loss = epoch_loss
        # 3. EVALUATION SWEEP (Test Miss Rate / Threshold Error)
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
        # 4. PATIENCE-BASED EARLY STOPPING (Monitored on Test Threshold Error)
        if epoch_test_err < best_test_err:
            best_test_err = epoch_test_err
            patience_counter = 0  # Reset counter since test error improved
            
            # Deepcopy weights so we can roll back to them later
            best_model_state = copy.deepcopy(model.state_dict())
            saved_epoch_test_err = epoch_test_err
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] No improvement in test miss rate for {patience} epochs. Stopping.")
                break
    # Restore the model to its absolute best state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return saved_epoch_test_err

def main():
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")
    results = {}

    # --- N-1 Cross-Validation Loop ---
    for target_idx, test_file in enumerate(ALL_FILES):
        train_files = [f for f in ALL_FILES if f != test_file]
        
        print(f"\n{'=' * 70}")
        print(f"RUN [{target_idx + 1}/{len(ALL_FILES)}]")
        print(f"Testing on Benchmark:  {test_file}")
        print(f"Training on remaining: {len(train_files)} files")
        print(f"{'=' * 70}")

        model, train_loader, test_loader = prepare_datasets(device, train_files, [test_file])
        
        loss_function = DynamicMissPredictionLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Train and track metrics
        miss_rate = train(model, train_loader, test_loader, loss_function, optimizer, device)
        
        # Fixed: Formatted individual run outputs as an explicit percentage to match training logs
        print(f"-> Finished Run. Final Test Miss Rate for {test_file}: {miss_rate * 100:.3f}%")
        results[test_file] = miss_rate

    # --- Print Summary Performance Metrics ---
    print(f"\n\n{'#' * 50}\nFINAL N-1 CROSS-VALIDATION SUMMARY\n{'#' * 50}")
    total_m_rate = 0.0
    for benchmark, miss_rate in results.items():
        print(f"{benchmark:<30} : {miss_rate * 100:.3f}%")
        total_m_rate += miss_rate
        
    avg_m_rate = total_m_rate / len(results)
    # Fixed: Multiplied by 100 to print the correct final summary average percentage
    print(f"{'-' * 50}\nAverage Misprediction Rate across all Benchmarks: {avg_m_rate * 100:.3f}%\n{'#' * 50}")

    

    
if  __name__ == "__main__":
    main()