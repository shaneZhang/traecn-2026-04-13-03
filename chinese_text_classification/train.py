import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import prepare_data, Vocabulary, LabelEncoder
from models.textcnn import TextCNN, TextCNNConfig


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='中文文本分类训练脚本')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--stopwords_path', type=str, default='data/stopwords.txt', help='停用词文件路径')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='模型输出目录')
    parser.add_argument('--embedding_dim', type=int, default=128, help='词向量维度')
    parser.add_argument('--num_filters', type=int, default=100, help='卷积核数量')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='卷积核大小')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--max_len', type=int, default=100, help='文本最大长度')
    parser.add_argument('--min_freq', type=int, default=2, help='最小词频')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, args.data_dir)
    stopwords_path = os.path.join(base_dir, args.stopwords_path)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("中文文本分类模型训练")
    print("=" * 50)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"词向量维度: {args.embedding_dim}")
    print(f"卷积核数量: {args.num_filters}")
    print(f"卷积核大小: {args.filter_sizes}")
    print(f"Dropout: {args.dropout}")
    print(f"学习率: {args.learning_rate}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print("=" * 50)
    
    print("\n正在准备数据...")
    train_dataset, val_dataset, test_dataset, vocab, label_encoder = prepare_data(
        data_dir, stopwords_path, args.max_len, args.min_freq
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"类别数量: {len(label_encoder)}")
    print(f"类别映射: {label_encoder.label2idx}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    
    config = TextCNNConfig(
        vocab_size=len(vocab),
        num_classes=len(label_encoder),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_sizes=filter_sizes,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_len=args.max_len
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = TextCNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        filter_sizes=config.filter_sizes,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print("\n开始训练...")
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config.to_dict()
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    print("\n在测试集上评估最佳模型...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"测试集损失: {test_loss:.4f}, 测试集准确率: {test_acc:.4f}")
    
    vocab.save(os.path.join(output_dir, 'vocab.pkl'))
    label_encoder.save(os.path.join(output_dir, 'label_encoder.pkl'))
    
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    print("\n训练完成！")
    print(f"模型已保存到: {output_dir}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")


if __name__ == "__main__":
    main()
