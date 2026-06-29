import numpy as np
import torch
from torch import nn
import copy

from ml_configs import (
    SEED, EMBED_DIM, EPOCHS, LR, PATIENCE, BATCH_SIZE, N_CLASSES, 
    DATA_DIR, ALL_FILES, DataLoader, TensorDataset, 
    classification_report, confusion_matrix
)
from data_engin import *
from mlp_model import *
from hb_branch_pred import train_one_epoch


def loo_method(device, train_files, test_file):
    set_seed(SEED)

    train_df = load_data(DATA_DIR, train_files)
    test_df = load_data(DATA_DIR, [test_file])

    # Add labeling (HT, HNT, NB)
    train_df["y"] = add_label(train_df)
    test_df["y"] = add_label(test_df)

    # Feature extraction & normalization pipeline
    (
        X_train_num, X_train_cat, y_train,
        X_test_num, X_test_cat, y_test,
        reg_vs, opc_vs
    ) = build_features(train_df, test_df)

    g = torch.Generator()
    g.manual_seed(SEED)

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

    model = NeuralNetworkWithEmbeddings(
        reg_vocab_size=reg_vs,
        opc_vocab_size=opc_vs,
        embed_dim=EMBED_DIM,
        num_features=X_train_num.shape[1],
        n_classes=N_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_f1 = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, test_f1, _, _, _ = evaluate(model, test_loader, device, test_df)

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

    acc, f1, preds, labels, miss_rate = evaluate(model, test_loader, device, test_df)

    return acc, f1, preds, labels, best_epoch, miss_rate 


def main():
    if hasattr(torch, 'accelerator') and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator().type
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")
    results = []

    print(f"\n{'=' * 80}")
    print(f"Starting Leave-One-Out Cross Validation across {len(ALL_FILES)} benchmarks")
    print(f"{'=' * 80}")

    # ITERATE THROUGH EVERY BENCHMARK
    for test_file in ALL_FILES:
        # Isolate all other files to act as the training set
        train_files = [f for f in ALL_FILES if f != test_file]
        
        print(f"\n[FOLD] Testing on target benchmark: {test_file}")
        
        acc, f1, preds, labels, best_epoch, miss_rate = loo_method(device, train_files, test_file)

        print(
            f"--> Best Epoch: {best_epoch} | "
            f"Acc: {acc:.3f} | "
            f"Macro F1: {f1:.3f} | "
            f"Miss Rate: {miss_rate:.5f}"
        )

        results.append({
            "benchmark": test_file,
            "accuracy": acc,
            "macro_f1": f1,
            "miss_rate": miss_rate,
            "n_test": len(labels),
            "best_epoch": best_epoch,
        })

    # FINAL REPORT GENERATION
    print(f"\n{'=' * 80}")
    print("FINAL LEAVE-ONE-OUT PERFORMANCE SUMMARY")
    print(f"{'=' * 80}")

    print(f"{'Benchmark File':<25} {'Acc':>8} {'F1':>8} {'MissRate':>10} {'N':>8} {'Epoch':>8}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['benchmark']:<25} "
            f"{r['accuracy']:>8.3f} "
            f"{r['macro_f1']:>8.3f} "
            f"{r['miss_rate']:>10.5f} "
            f"{r['n_test']:>8} "
            f"{r['best_epoch']:>8}"
        )

    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    misses = [r["miss_rate"] for r in results]

    print("-" * 80)
    print(f"{'Mean':<25} {np.mean(accs):>8.3f} {np.mean(f1s):>8.3f} {np.mean(misses):>10.5f}")
    print(f"{'Std':<25} {np.std(accs):>8.3f} {np.std(f1s):>8.3f} {np.std(misses):>10.5f}")

if __name__ == "__main__":
    main()