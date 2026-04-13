import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import os
import numpy as np

def plot_fold_metrics(history, labels_test, preds_test, probs_test, fold_name, save_path):
    """
    Plots Train/Val Loss, Confusion Matrix, ROC Curve, and PR Curve for a single fold or holdout.
    Saves the figure in the provided save_path
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{fold_name} - Curve di Loss")
    axes[0].legend()
    
    cm = confusion_matrix(labels_test, preds_test)
    ConfusionMatrixDisplay(cm).plot(ax=axes[1], cmap="Blues")
    axes[1].set_title(f"{fold_name} - Confusion Matrix")
    
    if len(np.unique(labels_test)) > 1:
        RocCurveDisplay.from_predictions(labels_test, probs_test, ax=axes[2])
        axes[2].set_title(f"{fold_name} - ROC Curve")
        PrecisionRecallDisplay.from_predictions(labels_test, probs_test, ax=axes[3])
        axes[3].set_title(f"{fold_name} - PR Curve")
    else:
        axes[2].text(0.5, 0.5, "Solo 1 classe", ha='center')
        axes[3].text(0.5, 0.5, "Solo 1 classe", ha='center')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_aggregated_metrics(labels_all, preds_all, probs_all, scheme_name, save_path):
    """
    Plots the aggregated Confusion Matrix, ROC, and PR Curves for the full CV.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_glob = confusion_matrix(labels_all, preds_all)
    ConfusionMatrixDisplay(cm_glob).plot(ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Aggregated Confusion Matrix ({scheme_name})")
    
    if len(np.unique(labels_all)) > 1:
        RocCurveDisplay.from_predictions(labels_all, probs_all, ax=axes[1])
        axes[1].set_title(f"Aggregated ROC Curve ({scheme_name})")
        PrecisionRecallDisplay.from_predictions(labels_all, probs_all, ax=axes[2])
        axes[2].set_title(f"Aggregated PR Curve ({scheme_name})")
    else:
        axes[1].text(0.5, 0.5, "Solo 1 classe", ha='center')
        axes[2].text(0.5, 0.5, "Solo 1 classe", ha='center')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
