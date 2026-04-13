import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int, 
                 filter_sizes: list, num_classes: int, dropout: float = 0.5,
                 pretrained_embeddings: torch.Tensor = None):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x)).squeeze(3)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pool_out)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


class TextCNNConfig:
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 128,
                 num_filters: int = 100, filter_sizes: list = None, 
                 dropout: float = 0.5, learning_rate: float = 0.001,
                 batch_size: int = 32, num_epochs: int = 20, max_len: int = 100):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_len = max_len
    
    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'max_len': self.max_len
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TextCNNConfig':
        return cls(**config_dict)
