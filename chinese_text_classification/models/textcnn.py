# -*- coding: utf-8 -*-
"""
TextCNN模型实现
用于中文文本分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN文本分类模型
    
    参数:
        vocab_size: 词表大小
        embedding_dim: 词向量维度
        num_classes: 类别数量
        num_filters: 卷积核数量
        filter_sizes: 卷积核尺寸列表
        dropout: Dropout概率
        pretrained_embedding: 预训练词向量 (可选)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_classes=5, 
                 num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5,
                 pretrained_embedding=None):
        super(TextCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 如果有预训练词向量，加载它
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
            self.embedding.weight.requires_grad = True  # 允许微调
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_length)
        
        返回:
            输出张量，形状为 (batch_size, num_classes)
        """
        # 词嵌入: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        x = self.embedding(x)
        
        # 添加通道维度: (batch_size, seq_length, embedding_dim) -> (batch_size, 1, seq_length, embedding_dim)
        x = x.unsqueeze(1)
        
        # 应用卷积和激活函数
        # 每个卷积输出形状: (batch_size, num_filters, seq_length - filter_size + 1, 1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        
        # 最大池化
        # 每个池化输出形状: (batch_size, num_filters, 1) -> (batch_size, num_filters)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        # 拼接所有卷积核的输出: (batch_size, len(filter_sizes) * num_filters)
        x = torch.cat(x, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # 全连接层: (batch_size, len(filter_sizes) * num_filters) -> (batch_size, num_classes)
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        """
        预测类别
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_length)
        
        返回:
            预测的类别索引，形状为 (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_model(vocab_size, num_classes, config):
    """
    创建TextCNN模型
    
    参数:
        vocab_size: 词表大小
        num_classes: 类别数量
        config: 配置字典
    
    返回:
        TextCNN模型实例
    """
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 128),
        num_classes=num_classes,
        num_filters=config.get('num_filters', 100),
        filter_sizes=config.get('filter_sizes', [3, 4, 5]),
        dropout=config.get('dropout', 0.5)
    )
    return model
