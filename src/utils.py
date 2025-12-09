import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, encoders, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'encoders': {k: v.classes_.tolist() for k, v in encoders.items()}}, path)


def load_checkpoint(model, path, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model_state'])
    return ck.get('encoders', None)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_history(history, outpath=None):
    # history: dict with train_loss/val_loss/train_auc/val_auc
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history['train_auc'], label='train_auc')
    plt.plot(history['val_auc'], label='val_auc')
    plt.legend(); plt.title('ROC-AUC')

    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()
