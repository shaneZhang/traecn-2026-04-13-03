import os
import sys
import argparse
import json
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import prepare_data, Vocabulary, LabelEncoder, load_stopwords, tokenize
from models.textcnn import TextCNN, TextCNNConfig


def compute_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    return confusion_matrix


def compute_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': confusion_matrix[i, :].sum()
        }
    
    total = confusion_matrix.sum()
    correct = np.diag(confusion_matrix).sum()
    accuracy = correct / total if total > 0 else 0
    
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    weighted_precision = sum(m['precision'] * m['support'] for m in metrics.values()) / total if total > 0 else 0
    weighted_recall = sum(m['recall'] * m['support'] for m in metrics.values()) / total if total > 0 else 0
    weighted_f1 = sum(m['f1'] * m['support'] for m in metrics.values()) / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted_avg': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'per_class': metrics
    }


def print_confusion_matrix(confusion_matrix, labels):
    print("\n混淆矩阵:")
    print("-" * 60)
    
    header = "真实\\预测"
    for label in labels:
        header += f"\t{label}"
    print(header)
    
    for i, label in enumerate(labels):
        row = f"{label}"
        for j in range(len(labels)):
            row += f"\t{confusion_matrix[i][j]}"
        print(row)
    print("-" * 60)


def print_classification_report(metrics, labels, label_encoder):
    print("\n分类报告:")
    print("-" * 80)
    print(f"{'类别':<10} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'支持数':<10}")
    print("-" * 80)
    
    for i in range(len(labels)):
        label = label_encoder.idx2label[i]
        m = metrics['per_class'][i]
        print(f"{label:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {int(m['support']):<10}")
    
    print("-" * 80)
    m = metrics['macro_avg']
    print(f"{'宏平均':<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    m = metrics['weighted_avg']
    print(f"{'加权平均':<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    print("-" * 80)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='中文文本分类评估脚本')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt', help='模型文件路径')
    parser.add_argument('--vocab_path', type=str, default='checkpoints/vocab.pkl', help='词汇表文件路径')
    parser.add_argument('--label_encoder_path', type=str, default='checkpoints/label_encoder.pkl', help='标签编码器路径')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--stopwords_path', type=str, default='data/stopwords.txt', help='停用词文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, args.model_path)
    vocab_path = os.path.join(base_dir, args.vocab_path)
    label_encoder_path = os.path.join(base_dir, args.label_encoder_path)
    data_dir = os.path.join(base_dir, args.data_dir)
    stopwords_path = os.path.join(base_dir, args.stopwords_path)
    
    print("=" * 50)
    print("中文文本分类模型评估")
    print("=" * 50)
    
    print("\n加载模型和配置...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = TextCNNConfig.from_dict(checkpoint['config'])
    
    vocab = Vocabulary.load(vocab_path)
    label_encoder = LabelEncoder.load(label_encoder_path)
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"类别数量: {len(label_encoder)}")
    print(f"类别映射: {label_encoder.label2idx}")
    
    model = TextCNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        filter_sizes=config.filter_sizes,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    print("\n加载测试数据...")
    stopwords = load_stopwords(stopwords_path)
    
    from utils.data_processor import load_data, TextDataset
    test_data = load_data(os.path.join(data_dir, 'test.json'))
    test_dataset = TextDataset(test_data, vocab, label_encoder, stopwords, config.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"测试集大小: {len(test_dataset)}")
    
    print("\n开始评估...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    num_classes = len(label_encoder)
    confusion_matrix = compute_confusion_matrix(all_labels, all_preds, num_classes)
    metrics = compute_metrics(confusion_matrix)
    
    labels = [label_encoder.idx2label[i] for i in range(num_classes)]
    
    print_confusion_matrix(confusion_matrix, labels)
    print_classification_report(metrics, labels, label_encoder)
    
    results = {
        'accuracy': metrics['accuracy'],
        'macro_avg': metrics['macro_avg'],
        'weighted_avg': metrics['weighted_avg'],
        'confusion_matrix': confusion_matrix.tolist(),
        'per_class': {
            label_encoder.idx2label[i]: {
                'precision': metrics['per_class'][i]['precision'],
                'recall': metrics['per_class'][i]['recall'],
                'f1': metrics['per_class'][i]['f1'],
                'support': int(metrics['per_class'][i]['support'])
            }
            for i in range(num_classes)
        }
    }
    
    results_path = os.path.join(os.path.dirname(model_path), 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {results_path}")


if __name__ == "__main__":
    main()
