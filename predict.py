import torch
import pickle
import jieba
import numpy as np
from gensim.models import Word2Vec

from config import Config
from models.textcnn import TextCNN
from utils.data_utils import load_stopwords, tokenize, text_to_sequence

config = Config()

def load_model():
    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    w2v_model = Word2Vec.load(config.w2v_path)
    
    from utils.data_utils import get_embedding_matrix
    embedding_matrix = get_embedding_matrix(w2v_model, vocab, config.embedding_dim)
    
    model = TextCNN(embedding_matrix).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    
    return model, vocab

def predict(text, model, vocab):
    stopwords = load_stopwords(config.stopwords_path)
    tokens = tokenize(text, stopwords)
    seq = text_to_sequence(tokens, vocab, config.max_len)
    
    x = torch.tensor([seq], dtype=torch.long).to(config.device)
    
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)
        pred_id = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][pred_id].item()
    
    label = config.id2label[pred_id]
    
    return {
        'label': label,
        'label_id': pred_id,
        'confidence': confidence,
        'probabilities': {
            config.id2label[i]: probabilities[0][i].item()
            for i in range(config.num_classes)
        }
    }

def main():
    print('Loading model...')
    model, vocab = load_model()
    print('Model loaded successfully!')
    print('=' * 60)
    
    test_texts = [
        '国家主席习近平出席金砖国家领导人会晤并发表重要讲话',
        '华为发布新一代麒麟芯片性能大幅提升',
        '中国女排夺得世界女排联赛冠军',
        '周杰伦发布最新专辑先行曲引发全网热议',
        '央行宣布降息0.25个百分点释放流动性'
    ]
    
    print('Running test predictions...')
    print('=' * 60)
    for text in test_texts:
        result = predict(text, model, vocab)
        print(f'\nText: {text}')
        print(f'Predicted: {result["label"]} (confidence: {result["confidence"]:.4f})')
        print('Probabilities:')
        for label, prob in result['probabilities'].items():
            print(f'  {label}: {prob:.4f}')
    print('=' * 60)
    
    print('\nInteractive prediction mode (enter "quit" to exit):')
    while True:
        text = input('\nPlease enter text to predict: ')
        if text.lower() in ['quit', 'exit', 'q']:
            break
        if not text.strip():
            continue
        
        result = predict(text, model, vocab)
        print(f'\nPredicted category: {result["label"]}')
        print(f'Confidence: {result["confidence"]:.4f}')
        print('\nAll category probabilities:')
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f'  {label:<6} {bar} {prob:.4f}')

if __name__ == '__main__':
    main()
