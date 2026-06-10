
from data_engin import *
from mlp_model import *
from ml_configs import SEED,EMBED_DIM,EPOCHS,LR,PATIENCE,BATCH_SIZE,N_CLASSES,DATA_DIR,TEST_FILES,TRAIN_FILES
from ml_configs import DataLoader,TensorDataset,copy,classification_report,confusion_matrix

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    n_samples = 0

    for x_num, x_cat, yb in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(x_num, x_cat)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_num.size(0)
        n_samples += x_num.size(0)

    return total_loss / n_samples




def run_model(device):
    set_seed(SEED)

    # Train and Test files
    train_df = load_data(DATA_DIR, TRAIN_FILES)
    test_df = load_data(DATA_DIR, TEST_FILES)

    # Total Executions of the test files
    
    
    # Add labeling (HT, HNT, NB)
    train_df["y"] = add_label(train_df)
    test_df["y"] = add_label(test_df)

    # Normalization of features
    (
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat, y_test,
        reg_vs, opc_vs
    ) = build_features(train_df, test_df)


    g = torch.Generator()
    g.manual_seed(SEED)

    # Load data to tensors (train, test)
    train_loader = DataLoader(
        TensorDataset(X_train_num, X_train_cat, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        TensorDataset(X_test_num, X_test_cat, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Create the model architecture
    model = NeuralNetworkWithEmbeddings(
        reg_vocab_size=reg_vs,
        opc_vocab_size=opc_vs,
        embed_dim=EMBED_DIM,
        num_features=X_train_num.shape[1],
        n_classes=N_CLASSES,
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Learning rate using Adam Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    # Train
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, test_f1, _, _,_ = evaluate(model, test_loader, device,test_df)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    acc, f1, preds, labels,miss_rate = evaluate(model, test_loader, device,test_df)

    

    return acc, f1, preds, labels, best_epoch,miss_rate 

def main():
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")
    results = []

    print(f"\n{'=' * 60}")
    print(f"Train files: {TRAIN_FILES}")
    print(f"Test files:  {TEST_FILES}")
    print(f"{'=' * 60}")

    acc, f1, preds, labels, best_epoch,miss_rate = run_model(device)

    print(
        f"\nBest epoch: {best_epoch}  "
        f"acc={acc:.3f}  "
        f"macro_f1={f1:.3f}"
        f"miss_rate={miss_rate:.5f}"
    )

    print(
        classification_report(
            labels.numpy(),
            preds.numpy(),
            target_names=["HNT", "NB", "HT"],
            digits=3,
        )
    )

    cm = confusion_matrix(labels.numpy(), preds.numpy())
    print(f"\nConfusion Matrix:\n{cm}")

    results.append({
        "accuracy": acc,
        "macro_f1": f1,
        "miss_rate": miss_rate,
        "n_test": len(labels),
        "best_epoch": best_epoch,
    })

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    print(f"{'Acc':>8} {'F1':>8} {'MissRate':>10} {'N':>8} {'Epoch':>8}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['accuracy']:>8.3f} "
            f"{r['macro_f1']:>8.3f} "
            f"{r['miss_rate']:>10.5f} "
            f"{r['n_test']:>8} "
            f"{r['best_epoch']:>8}"
        )

    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    misses = [r["miss_rate"] for r in results]

    print("-" * 70)
    print("-" * 80)
    print(f"{'Mean':<30} {np.mean(accs):>8.3f} {np.mean(f1s):>8.3f} {np.mean(misses):>10.5f}")
    print(f"{'Std':<30} {np.std(accs):>8.3f} {np.std(f1s):>8.3f} {np.std(misses):>10.5f}")


if __name__ == "__main__":
    main()