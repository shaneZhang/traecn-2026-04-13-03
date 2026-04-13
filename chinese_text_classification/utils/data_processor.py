# -*- coding: utf-8 -*-
"""
数据预处理模块
包括：文本分词、去除停用词、构建词表、数据加载
"""

import os
import re
import pickle
import jieba
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class TextProcessor:
    """文本处理器：分词、去停用词、序列化"""
    
    def __init__(self, stopwords_path=None, max_vocab_size=10000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.stopwords = set()
        
        # 加载停用词
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
    
    def tokenize(self, text):
        """分词并去除停用词"""
        # 去除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词和空字符串
        words = [w.strip() for w in words if w.strip() and w.strip() not in self.stopwords and len(w.strip()) > 1]
        return words
    
    def build_vocab(self, texts):
        """构建词表"""
        word_counter = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_counter.update(words)
        
        # 选择频率最高的词
        most_common = word_counter.most_common(self.max_vocab_size - 2)  # 保留PAD和UNK的位置
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词表大小: {len(self.word2idx)}")
        return self.word2idx
    
    def text_to_sequence(self, text):
        """将文本转换为索引序列"""
        words = self.tokenize(text)
        sequence = [self.word2idx.get(word, 1) for word in words]  # 1是<UNK>的索引
        
        # 填充或截断
        if len(sequence) < self.max_seq_length:
            sequence = sequence + [0] * (self.max_seq_length - len(sequence))  # 0是<PAD>的索引
        else:
            sequence = sequence[:self.max_seq_length]
        
        return sequence
    
    def save_vocab(self, path):
        """保存词表"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'max_seq_length': self.max_seq_length
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocab(self, path):
        """加载词表"""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.max_seq_length = vocab_data['max_seq_length']


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts, labels, text_processor, label2idx=None):
        self.texts = texts
        self.labels = labels
        self.text_processor = text_processor
        
        # 构建标签映射
        if label2idx is None:
            unique_labels = sorted(list(set(labels)))
            self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx
        
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
        # 预处理所有文本
        self.sequences = [self.text_processor.text_to_sequence(text) for text in texts]
        self.label_indices = [self.label2idx[label] for label in labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.label_indices[idx], dtype=torch.long)
        }
    
    def get_label_names(self):
        """获取标签名称列表"""
        return [self.idx2label[i] for i in range(len(self.idx2label))]


def load_data(csv_path):
    """从CSV加载数据"""
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def split_data(texts, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """划分训练集、验证集、测试集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(texts))
    
    train_size = int(len(texts) * train_ratio)
    val_size = int(len(texts) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def create_data_loaders(train_data, val_data, test_data, text_processor, batch_size=32):
    """创建数据加载器"""
    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data
    
    # 创建数据集
    train_dataset = TextDataset(train_texts, train_labels, text_processor)
    val_dataset = TextDataset(val_texts, val_labels, text_processor, label2idx=train_dataset.label2idx)
    test_dataset = TextDataset(test_texts, test_labels, text_processor, label2idx=train_dataset.label2idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.label2idx
