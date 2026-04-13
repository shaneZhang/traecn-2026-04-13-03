import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from config import Config

config = Config()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def print_metrics(metrics):
    print(f'\nOverall Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Macro Precision: {metrics["precision_macro"]:.4f}')
    print(f'Macro Recall: {metrics["recall_macro"]:.4f}')
    print(f'Macro F1: {metrics["f1_macro"]:.4f}')
    
    print('\nPer-class metrics:')
    print('-' * 60)
    print(f'{"Class":<10} {"Precision":<12} {"Recall":<12} {"F1":<12}')
    print('-' * 60)
    
    for i, label in config.id2label.items():
        print(f'{label:<10} {metrics["precision"][i]:<12.4f} {metrics["recall"][i]:<12.4f} {metrics["f1"][i]:<12.4f}')
    print('-' * 60)

def plot_confusion_matrix(y_true, y_pred, save_path='data/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    labels = [config.id2label[i] for i in range(config.num_classes)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\nConfusion matrix saved to {save_path}')
