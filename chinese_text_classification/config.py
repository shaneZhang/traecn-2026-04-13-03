# -*- coding: utf-8 -*-
"""
项目配置文件
"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据配置
DATA_CONFIG = {
    'data_path': os.path.join(BASE_DIR, 'data', 'news_data.csv'),
    'stopwords_path': os.path.join(BASE_DIR, 'stopwords', 'stopwords.txt'),
    'vocab_path': os.path.join(BASE_DIR, 'models', 'vocab.pkl'),
    'samples_per_category': 200,  # 每个类别的样本数
}

# 模型配置
MODEL_CONFIG = {
    'embedding_dim': 128,        # 词向量维度
    'num_filters': 100,          # 卷积核数量
    'filter_sizes': [3, 4, 5],   # 卷积核尺寸
    'dropout': 0.5,              # Dropout概率
    'max_vocab_size': 5000,      # 最大词表大小
    'max_seq_length': 50,        # 最大序列长度
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,            # 批次大小
    'num_epochs': 20,            # 训练轮数
    'learning_rate': 0.001,      # 学习率
    'weight_decay': 1e-5,        # 权重衰减
    'train_ratio': 0.7,          # 训练集比例
    'val_ratio': 0.15,           # 验证集比例
    'test_ratio': 0.15,          # 测试集比例
    'random_seed': 42,           # 随机种子
    'checkpoint_dir': os.path.join(BASE_DIR, 'models', 'checkpoints'),
    'best_model_path': os.path.join(BASE_DIR, 'models', 'best_model.pth'),
}

# 评估配置
EVAL_CONFIG = {
    'plot_confusion_matrix': True,
    'save_confusion_matrix': os.path.join(BASE_DIR, 'results', 'confusion_matrix.png'),
}

# 类别配置
CATEGORIES = ['政治', '科技', '体育', '娱乐', '财经']

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': True,  # 是否使用GPU
}


def get_config():
    """获取完整配置"""
    config = {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'eval': EVAL_CONFIG,
        'categories': CATEGORIES,
        'device': DEVICE_CONFIG,
    }
    return config
