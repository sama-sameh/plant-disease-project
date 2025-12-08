import os
import copy
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score, f1_score
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing
import sys
import json

# ---------- SETTINGS ----------
DATA_DIR = r"C:\Users\hp\Downloads\plantvillage_dataset_New\plantvillage_dataset_New"
OUT_DIR = r"C:\Users\hp\Downloads\plantvillage_dataset_New\plantvillage_dataset_New\outputs"

BATCH_SIZE = 32
NUM_WORKERS = 6     # set to 0 on Windows if you encounter freezes
NUM_EPOCHS_HEAD = 6
NUM_EPOCHS_FINETUNE = 12
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -----------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------
def get_transforms(img_size=IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return train_tf, test_tf

# -----------------------------------------------------------
# GoogLeNet (Inception v1) builder
# -----------------------------------------------------------
def build_googlenet(num_classes, device, aux_logits=True):
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=aux_logits)
    # replace final fc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model.aux_logits:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    return model.to(device)

# -----------------------------------------------------------
# Train one epoch
# -----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        if isinstance(out, tuple):
            main, a1, a2 = out
            loss = criterion(main, y) + 0.3 * criterion(a1, y) + 0.3 * criterion(a2, y)
        else:
            main = out
            loss = criterion(main, y)

        loss.backward()
        optimizer.step()

        preds = main.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / total, correct / total

# -----------------------------------------------------------
# Eval (returns loss, acc, y_true, y_pred, y_probs)
# -----------------------------------------------------------
def eval_model_with_probs(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            main = out[0] if isinstance(out, tuple) else out

            loss = criterion(main, y)
            probs = torch.softmax(main, dim=1)

            preds = main.argmax(1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.vstack(all_probs)

    return total_loss / total, (all_preds == all_labels).mean(), all_labels, all_preds, all_probs

# -----------------------------------------------------------
# Utilities: plotting and saving
# -----------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12,10))
    sns.heatmap(df, annot=False, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_training_curves(history_df, out_prefix):
    train = history_df[history_df.phase == "train"]
    val   = history_df[history_df.phase == "val"]

    plt.figure()
    plt.plot(train.epoch, train.loss, label="train_loss")
    plt.plot(val.epoch, val.loss, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Loss vs Epoch")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(train.epoch, train.acc, label="train_acc")
    plt.plot(val.epoch, val.acc, label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Accuracy vs Epoch")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_acc.png", dpi=200)
    plt.close()

def plot_roc_pr(y_true, y_probs, class_names, out_prefix):
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # ROC per class
    plt.figure(figsize=(8,6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc_per_class.png", dpi=200)
    plt.close()

    # Macro AUC
    try:
        macro_auc = roc_auc_score(y_bin, y_probs, average="macro")
    except Exception:
        macro_auc = None

    # Precision-Recall per class
    plt.figure(figsize=(8,6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:,i], y_probs[:,i])
        ap = average_precision_score(y_bin[:,i], y_probs[:,i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr_per_class.png", dpi=200)
    plt.close()

    return macro_auc

# -----------------------------------------------------------
# Pipeline for GoogLeNet only (head + finetune)
# -----------------------------------------------------------
def train_and_eval_googlenet(dataloaders, class_weights, class_names, device):
    train_loader, val_loader, test_loader = dataloaders
    num_classes = len(class_names)

    model = build_googlenet(num_classes, device, aux_logits=True)

    # criterion expects weight on same device as model params
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ---- TRAIN HEAD ----
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True
    # enable aux classifiers' params
    if hasattr(model, "aux1"):
        for p in model.aux1.parameters():
            p.requires_grad = True
        for p in model.aux2.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    history = {"epoch": [], "phase": [], "loss": [], "acc": []}
    best_acc = 0
    best_wts = copy.deepcopy(model.state_dict())

    print("=== Training HEAD (GoogLeNet) ===")
    for epoch in range(NUM_EPOCHS_HEAD):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_HEAD}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = eval_model_with_probs(model, val_loader, criterion, device)

        print(f"Train {tr_acc:.4f} | Val {val_acc:.4f}")
        scheduler.step(val_acc)

        # save history
        history["epoch"].append(epoch+1); history["phase"].append("train"); history["loss"].append(tr_loss); history["acc"].append(tr_acc)
        history["epoch"].append(epoch+1); history["phase"].append("val"); history["loss"].append(val_loss); history["acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(OUT_DIR, "googlenet_best_head.pth"))

    # ---- FINE-TUNE ALL ----
    print("=== Fine-tuning ALL (GoogLeNet) ===")
    model.load_state_dict(best_wts)  # start finetune from best head
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), lr=LR_FINETUNE, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    for epoch in range(NUM_EPOCHS_FINETUNE):
        print(f"Fine-tune {epoch+1}/{NUM_EPOCHS_FINETUNE}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = eval_model_with_probs(model, val_loader, criterion, device)

        print(f"Train {tr_acc:.4f} | Val {val_acc:.4f}")
        scheduler.step(val_acc)

        history["epoch"].append(NUM_EPOCHS_HEAD + epoch+1); history["phase"].append("train"); history["loss"].append(tr_loss); history["acc"].append(tr_acc)
        history["epoch"].append(NUM_EPOCHS_HEAD + epoch+1); history["phase"].append("val"); history["loss"].append(val_loss); history["acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(OUT_DIR, "googlenet_best_finetuned.pth"))

    # load best and save final
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "googlenet_best_overall.pth"))

    # ---- TEST ----
    test_loss, test_acc, y_true, y_pred, y_probs = eval_model_with_probs(model, test_loader, criterion, device)
    print("TEST ACC:", test_acc)

    # classification report
    creport = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(creport)

    # save classification report
    with open(os.path.join(OUT_DIR, "googlenet_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(creport)

    # confusion matrix
    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(OUT_DIR, "googlenet_confusion_matrix.png"))

    # save raw preds & probs
    np.savez(os.path.join(OUT_DIR, "googlenet_test_preds.npz"), y_true=y_true, y_pred=y_pred, y_probs=y_probs)

    # plots: ROC & PR
    macro_auc = plot_roc_pr(y_true, y_probs, class_names, os.path.join(OUT_DIR, "googlenet"))

    # history -> dataframe + plots
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUT_DIR, "googlenet_training_history.csv"), index=False)
    plot_training_curves(hist_df, os.path.join(OUT_DIR, "googlenet"))

    # summary metrics
    summary = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "num_classes": len(class_names),
        "class_names": class_names,
        "macro_auc": float(macro_auc) if macro_auc is not None else None
    }
    with open(os.path.join(OUT_DIR, "googlenet_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_tf, test_tf = get_transforms()

    # Load data
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=test_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

    print("Classes:", train_ds.classes)
    class_names = train_ds.classes
    num_classes = len(class_names)

    # -------- CLASS WEIGHTS --------
    class_counts = np.zeros(num_classes)
    for _, label in train_ds.samples:
        class_counts[label] += 1

    # avoid zero division
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = torch.tensor(1.0 / np.sqrt(class_counts), dtype=torch.float32)

    print("\n=== CLASS WEIGHTS ===")
    print(class_weights)

    # -------- DATALOADERS --------
    pin_memory = True if device == "cuda" else False
    effective_workers = NUM_WORKERS if NUM_WORKERS > 0 else 0

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=effective_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=effective_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=effective_workers, pin_memory=pin_memory)

    # ---- Train & Eval GoogLeNet only ----
    summary = train_and_eval_googlenet((train_loader, val_loader, test_loader),
                                       class_weights, class_names, device)

    # Save metadata & README
    metadata = {
        "device": device,
        "pytorch_version": torch.__version__,
        "torchvision_version": getattr(models, "__version__", "unknown"),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "num_epochs_head": NUM_EPOCHS_HEAD,
        "num_epochs_finetune": NUM_EPOCHS_FINETUNE,
        "lr_head": LR_HEAD,
        "lr_finetune": LR_FINETUNE,
        "img_size": IMG_SIZE,
        "class_weights": class_weights.tolist(),
        "summary": summary
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""
GoogLeNet (Inception-V1) transfer learning run
---------------------------------------------
Data dir: {DATA_DIR}
Output dir: {OUT_DIR}

Model: GoogLeNet pretrained on ImageNet
Training strategy:
 - Train head for {NUM_EPOCHS_HEAD} epochs (AdamW lr={LR_HEAD})
 - Fine-tune all weights for {NUM_EPOCHS_FINETUNE} epochs (SGD lr={LR_FINETUNE})
 - Class weighting applied using 1/sqrt(class_count)

Saved files (examples):
 - googlenet_best_head.pth
 - googlenet_best_finetuned.pth
 - googlenet_best_overall.pth
 - googlenet_training_history.csv
 - googlenet_confusion_matrix.png
 - googlenet_roc_per_class.png
 - googlenet_pr_per_class.png
 - googlenet_acc.png, googlenet_loss.png
 - googlenet_classification_report.txt
 - googlenet_summary.json
 - metadata.json

Notes / Pros & Cons:
 - Pros: Fast convergence using pretrained features; auxiliary classifiers in GoogLeNet help gradients.
 - Cons: GoogLeNet is older and may be less accurate than newer architectures; if you need highest accuracy consider ResNet/EfficientNet variants.
 - If you see dataloader hangs on Windows, set NUM_WORKERS=0.

"""
    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("All done. Outputs saved to", OUT_DIR)

if __name__ == "__main__":
    main()
