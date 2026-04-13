import jieba
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

config = Config()

def load_stopwords(path):
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def tokenize(text, stopwords):
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords and w.strip()]
    return words

def build_vocab(texts, min_freq=2):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_counts = {}
    
    for text in texts:
        for word in text:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def text_to_sequence(text, vocab, max_len):
    seq = [vocab.get(word, 1) for word in text]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

def train_word2vec(texts, embedding_dim, save_path):
    model = Word2Vec(
        sentences=texts,
        vector_size=embedding_dim,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )
    model.save(save_path)
    return model

def get_embedding_matrix(w2v_model, vocab, embedding_dim):
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim))
    
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    
    return embedding_matrix

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def load_data():
    print('Loading data...')
    df = pd.read_csv(config.data_path)
    stopwords = load_stopwords(config.stopwords_path)
    
    print('Tokenizing...')
    tqdm.pandas()
    df['tokens'] = df['text'].progress_apply(lambda x: tokenize(x, stopwords))
    
    df['label_id'] = df['label'].map(config.label_map)
    
    print('Building vocabulary...')
    vocab = build_vocab(df['tokens'].tolist())
    with open(config.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Vocabulary size: {len(vocab)}')
    
    print('Training Word2Vec...')
    w2v_model = train_word2vec(df['tokens'].tolist(), config.embedding_dim, config.w2v_path)
    embedding_matrix = get_embedding_matrix(w2v_model, vocab, config.embedding_dim)
    
    print('Converting texts to sequences...')
    df['seq'] = df['tokens'].apply(lambda x: text_to_sequence(x, vocab, config.max_len))
    
    X = df['seq'].tolist()
    y = df['label_id'].tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - config.train_size), random_state=config.seed, stratify=y
    )
    val_ratio = config.val_size / (config.val_size + config.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=config.seed, stratify=y_temp
    )
    
    print(f'Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}')
    
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    test_dataset = TextDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    return train_loader, val_loader, test_loader, embedding_matrix, vocab
