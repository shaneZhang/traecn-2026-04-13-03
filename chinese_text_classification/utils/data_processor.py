import json
import os
import pickle
from collections import Counter
from typing import List, Dict, Tuple

import jieba
import numpy as np
import torch
from torch.utils.data import Dataset


def load_stopwords(stopwords_path: str) -> set:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    return stopwords


def tokenize(text: str, stopwords: set = None) -> List[str]:
    words = jieba.lcut(text)
    if stopwords:
        words = [w for w in words if w not in stopwords and len(w.strip()) > 0]
    return words


def load_data(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        
    def build_vocab(self, texts: List[List[str]]):
        for text in texts:
            self.word_freq.update(text)
        
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        
        idx = 2
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, words: List[str]) -> List[int]:
        return [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in words]
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_freq': self.min_freq
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']
        return vocab


class LabelEncoder:
    def __init__(self):
        self.label2idx = {}
        self.idx2label = {}
        
    def fit(self, labels: List[str]):
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
    def encode(self, label: str) -> int:
        return self.label2idx[label]
    
    def decode(self, idx: int) -> str:
        return self.idx2label[idx]
    
    def __len__(self):
        return len(self.label2idx)
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label2idx': self.label2idx,
                'idx2label': self.idx2label
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'LabelEncoder':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        encoder = cls()
        encoder.label2idx = data['label2idx']
        encoder.idx2label = data['idx2label']
        return encoder


class TextDataset(Dataset):
    def __init__(self, data: List[Dict], vocab: Vocabulary, label_encoder: LabelEncoder,
                 stopwords: set, max_len: int = 100):
        self.data = data
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.stopwords = stopwords
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        words = tokenize(text, self.stopwords)
        word_ids = self.vocab.encode(words)
        
        if len(word_ids) < self.max_len:
            word_ids = word_ids + [0] * (self.max_len - len(word_ids))
        else:
            word_ids = word_ids[:self.max_len]
        
        label_id = self.label_encoder.encode(label)
        
        return {
            'text': torch.tensor(word_ids, dtype=torch.long),
            'label': torch.tensor(label_id, dtype=torch.long),
            'raw_text': text
        }


def prepare_data(data_dir: str, stopwords_path: str, max_len: int = 100, 
                 min_freq: int = 2) -> Tuple[TextDataset, TextDataset, TextDataset, 
                                              Vocabulary, LabelEncoder]:
    stopwords = load_stopwords(stopwords_path)
    
    train_data = load_data(os.path.join(data_dir, 'train.json'))
    val_data = load_data(os.path.join(data_dir, 'val.json'))
    test_data = load_data(os.path.join(data_dir, 'test.json'))
    
    all_data = train_data + val_data + test_data
    all_texts = [tokenize(item['text'], stopwords) for item in all_data]
    all_labels = [item['label'] for item in all_data]
    
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(all_texts)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    train_dataset = TextDataset(train_data, vocab, label_encoder, stopwords, max_len)
    val_dataset = TextDataset(val_data, vocab, label_encoder, stopwords, max_len)
    test_dataset = TextDataset(test_data, vocab, label_encoder, stopwords, max_len)
    
    return train_dataset, val_dataset, test_dataset, vocab, label_encoder


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    stopwords_path = os.path.join(data_dir, 'stopwords.txt')
    
    train_dataset, val_dataset, test_dataset, vocab, label_encoder = prepare_data(
        data_dir, stopwords_path
    )
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"类别数量: {len(label_encoder)}")
    print(f"类别映射: {label_encoder.label2idx}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    sample = train_dataset[0]
    print(f"\n样本示例:")
    print(f"原文: {sample['raw_text']}")
    print(f"编码后: {sample['text'][:20]}...")
    print(f"标签: {sample['label']}")
