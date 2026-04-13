# -*- coding: utf-8 -*-
"""
训练脚本
完整的中文文本分类训练流程
"""

import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.textcnn import create_model
from utils.data_processor import TextProcessor, load_data, split_data, create_data_loaders
from utils.trainer import Trainer
from utils.evaluator import Evaluator, evaluate_model
from utils.generate_data import generate_dataset


def setup_directories(config):
    """创建必要的目录"""
    os.makedirs(os.path.dirname(config['data']['data_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['vocab_path']), exist_ok=True)
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['eval']['save_confusion_matrix']), exist_ok=True)


def plot_training_history(history, save_path=None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("中文文本分类 - 模型训练")
    print("=" * 60)
    
    # 加载配置
    config = get_config()
    setup_directories(config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 步骤1: 生成数据
    print("\n" + "-" * 60)
    print("步骤1: 生成数据集")
    print("-" * 60)
    if not os.path.exists(config['data']['data_path']):
        generate_dataset(
            config['data']['data_path'],
            samples_per_category=config['data']['samples_per_category']
        )
    else:
        print(f"数据集已存在: {config['data']['data_path']}")
    
    # 步骤2: 加载和划分数据
    print("\n" + "-" * 60)
    print("步骤2: 加载和划分数据")
    print("-" * 60)
    texts, labels = load_data(config['data']['data_path'])
    print(f"总样本数: {len(texts)}")
    
    train_data, val_data, test_data = split_data(
        texts, labels,
        train_ratio=config['train']['train_ratio'],
        val_ratio=config['train']['val_ratio'],
        test_ratio=config['train']['test_ratio'],
        random_seed=config['train']['random_seed']
    )
    print(f"训练集: {len(train_data[0])}条")
    print(f"验证集: {len(val_data[0])}条")
    print(f"测试集: {len(test_data[0])}条")
    
    # 步骤3: 构建词表
    print("\n" + "-" * 60)
    print("步骤3: 构建词表")
    print("-" * 60)
    text_processor = TextProcessor(
        stopwords_path=config['data']['stopwords_path'],
        max_vocab_size=config['model']['max_vocab_size'],
        max_seq_length=config['model']['max_seq_length']
    )
    text_processor.build_vocab(train_data[0])
    text_processor.save_vocab(config['data']['vocab_path'])
    print(f"词表已保存至: {config['data']['vocab_path']}")
    
    # 步骤4: 创建数据加载器
    print("\n" + "-" * 60)
    print("步骤4: 创建数据加载器")
    print("-" * 60)
    train_loader, val_loader, test_loader, label2idx = create_data_loaders(
        train_data, val_data, test_data,
        text_processor,
        batch_size=config['train']['batch_size']
    )
    print(f"类别映射: {label2idx}")
    
    # 步骤5: 创建模型
    print("\n" + "-" * 60)
    print("步骤5: 创建模型")
    print("-" * 60)
    vocab_size = len(text_processor.word2idx)
    num_classes = len(label2idx)
    model = create_model(vocab_size, num_classes, config['model'])
    print(f"词表大小: {vocab_size}")
    print(f"类别数量: {num_classes}")
    print(f"模型结构:\n{model}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 步骤6: 训练模型
    print("\n" + "-" * 60)
    print("步骤6: 训练模型")
    print("-" * 60)
    trainer = Trainer(model, device, config['train'])
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=config['train']['num_epochs']
    )
    
    # 保存最终模型
    final_model_path = config['train']['best_model_path']
    torch.save({
        'model_state_dict': model.state_dict(),
        'label2idx': label2idx,
        'config': config
    }, final_model_path)
    print(f"\n最终模型已保存至: {final_model_path}")
    
    # 绘制训练历史
    history_path = os.path.join(os.path.dirname(config['eval']['save_confusion_matrix']), 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # 步骤7: 评估模型
    print("\n" + "-" * 60)
    print("步骤7: 评估模型")
    print("-" * 60)
    label_names = [label for label, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    metrics = evaluate_model(
        model, test_loader, device, label_names,
        plot_cm=config['eval']['plot_confusion_matrix'],
        save_cm_path=config['eval']['save_confusion_matrix']
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳验证准确率: {trainer.best_val_acc:.2f}%")
    print(f"测试集准确率: {metrics['accuracy']*100:.2f}%")
    print(f"模型保存路径: {final_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
