try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class Config:
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        
        self.data_path = 'data/news_dataset.csv'
        self.stopwords_path = 'data/stopwords.txt'
        self.vocab_path = 'data/vocab.pkl'
        self.w2v_path = 'data/word2vec.model'
        
        self.save_path = 'models/textcnn_model.pth'
        
        self.seed = 42
        self.max_len = 100
        self.embedding_dim = 100
        self.num_filters = 64
        self.filter_sizes = [2, 3, 4]
        self.dropout = 0.5
        self.num_classes = 5
        
        self.batch_size = 32
        self.epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-4
        
        self.train_size = 0.7
        self.val_size = 0.15
        self.test_size = 0.15
        
        self.label_map = {
            '政治': 0,
            '科技': 1,
            '体育': 2,
            '娱乐': 3,
            '财经': 4
        }
        self.id2label = {v: k for k, v in self.label_map.items()}
