import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import Vocabulary, LabelEncoder, load_stopwords, tokenize
from models.textcnn import TextCNN, TextCNNConfig


class TextClassifier:
    def __init__(self, model_path: str, vocab_path: str, label_encoder_path: str, 
                 stopwords_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location='cpu')
        self.config = TextCNNConfig.from_dict(checkpoint['config'])
        
        self.vocab = Vocabulary.load(vocab_path)
        self.label_encoder = LabelEncoder.load(label_encoder_path)
        
        self.model = TextCNN(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            num_filters=self.config.num_filters,
            filter_sizes=self.config.filter_sizes,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            self.stopwords = load_stopwords(stopwords_path)
    
    def predict(self, text: str) -> tuple:
        words = tokenize(text, self.stopwords)
        word_ids = self.vocab.encode(words)
        
        if len(word_ids) < self.config.max_len:
            word_ids = word_ids + [0] * (self.config.max_len - len(word_ids))
        else:
            word_ids = word_ids[:self.config.max_len]
        
        input_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_label = self.label_encoder.decode(predicted_class)
        
        all_probs = {
            self.label_encoder.decode(i): probabilities[0][i].item()
            for i in range(len(self.label_encoder))
        }
        
        return predicted_label, confidence, all_probs
    
    def predict_batch(self, texts: list) -> list:
        results = []
        for text in texts:
            label, conf, probs = self.predict(text)
            results.append({
                'text': text,
                'predicted_label': label,
                'confidence': conf,
                'probabilities': probs
            })
        return results


def main():
    parser = argparse.ArgumentParser(description='中文文本分类预测脚本')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt', help='模型文件路径')
    parser.add_argument('--vocab_path', type=str, default='checkpoints/vocab.pkl', help='词汇表文件路径')
    parser.add_argument('--label_encoder_path', type=str, default='checkpoints/label_encoder.pkl', help='标签编码器路径')
    parser.add_argument('--stopwords_path', type=str, default='data/stopwords.txt', help='停用词文件路径')
    parser.add_argument('--text', type=str, default=None, help='要预测的文本')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, args.model_path)
    vocab_path = os.path.join(base_dir, args.vocab_path)
    label_encoder_path = os.path.join(base_dir, args.label_encoder_path)
    stopwords_path = os.path.join(base_dir, args.stopwords_path) if args.stopwords_path else None
    
    print("=" * 50)
    print("中文文本分类预测")
    print("=" * 50)
    
    print("\n加载模型...")
    classifier = TextClassifier(model_path, vocab_path, label_encoder_path, stopwords_path)
    print(f"使用设备: {classifier.device}")
    print(f"类别: {list(classifier.label_encoder.label2idx.keys())}")
    
    if args.text:
        print(f"\n输入文本: {args.text}")
        label, confidence, probs = classifier.predict(args.text)
        print(f"\n预测结果: {label}")
        print(f"置信度: {confidence:.4f}")
        print("\n各类别概率:")
        for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {prob:.4f}")
    
    elif args.interactive:
        print("\n进入交互模式，输入 'quit' 或 'exit' 退出")
        print("-" * 50)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("退出程序")
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                label, confidence, probs = classifier.predict(text)
                
                print(f"\n预测结果: {label}")
                print(f"置信度: {confidence:.4f}")
                print("各类别概率:")
                for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cat}: {prob:.4f}")
                    
            except KeyboardInterrupt:
                print("\n退出程序")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    else:
        test_texts = [
            "国务院召开会议研究部署经济工作，强调要深入贯彻落实科学发展观",
            "我国成功发射新一代卫星，标志着航天事业取得新突破",
            "中国队在世界杯比赛中获得冠军，创造历史最好成绩",
            "电影票房突破十亿，成为年度黑马",
            "央行宣布下调存款准备金率，A股市场集体上涨"
        ]
        
        print("\n示例预测:")
        print("-" * 50)
        
        for text in test_texts:
            label, confidence, probs = classifier.predict(text)
            print(f"\n文本: {text}")
            print(f"预测: {label} (置信度: {confidence:.4f})")


if __name__ == "__main__":
    main()
