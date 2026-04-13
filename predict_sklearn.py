import pickle
import jieba
from config import Config

config = Config()

def load_stopwords(path):
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def tokenize(text):
    stopwords = load_stopwords(config.stopwords_path)
    words = jieba.lcut(text)
    words = [w for w in words if w not in stopwords and w.strip()]
    return ' '.join(words)

def load_model():
    with open('data/sklearn_model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def predict(text, model, vectorizer):
    tokens = tokenize(text)
    vec = vectorizer.transform([tokens])
    pred_id = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    
    label = config.id2label[pred_id]
    confidence = probs[pred_id]
    
    return {
        'label': label,
        'label_id': pred_id,
        'confidence': confidence,
        'probabilities': {
            config.id2label[i]: probs[i]
            for i in range(config.num_classes)
        }
    }

def main():
    print('=' * 60)
    print('中文文本分类预测系统 (scikit-learn版本)')
    print('=' * 60)
    
    print('\nLoading model...')
    model, vectorizer = load_model()
    print('Model loaded successfully!')
    
    print('\n' + '=' * 60)
    print('内置测试:')
    print('=' * 60)
    
    test_texts = [
        '国务院发布关于加强数字政府建设的指导意见',
        '苹果发布最新款iPhone手机搭载A17芯片',
        '梅西夺得世界杯冠军加冕球王',
        '赵丽颖出演古装剧获得收视佳绩',
        'A股三大指数集体上涨沪指收复3000点'
    ]
    
    for text in test_texts:
        result = predict(text, model, vectorizer)
        print(f'\n文本: {text}')
        print(f'预测: {result["label"]} (置信度: {result["confidence"]:.4f})')
    
    print('\n' + '=' * 60)
    print('交互式预测模式 (输入 quit 退出)')
    print('=' * 60)
    
    while True:
        text = input('\n请输入要预测的文本: ')
        if text.lower() in ['quit', 'exit', 'q', '退出']:
            break
        if not text.strip():
            continue
        
        result = predict(text, model, vectorizer)
        print(f'\n预测类别: {result["label"]}')
        print(f'置信度: {result["confidence"]:.4f}')
        print('\n各类别概率:')
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f'  {label:<6} {bar} {prob:.4f}')

if __name__ == '__main__':
    main()
