# -*- coding: utf-8 -*-
"""
预测模块
用于对新文本进行分类预测
"""

import torch
import pickle


class Predictor:
    """文本分类预测器"""
    
    def __init__(self, model, text_processor, label2idx, device='cpu'):
        """
        参数:
            model: 训练好的模型
            text_processor: 文本处理器
            label2idx: 标签到索引的映射
            device: 计算设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.text_processor = text_processor
        self.device = device
        
        # 创建索引到标签的映射
        self.idx2label = {idx: label for label, idx in label2idx.items()}
        self.labels = [self.idx2label[i] for i in range(len(self.idx2label))]
    
    def predict(self, text):
        """
        预测单条文本的类别
        
        参数:
            text: 输入文本字符串
        
        返回:
            预测结果字典，包含类别、概率等
        """
        # 文本转序列
        sequence = self.text_processor.text_to_sequence(text)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # 获取所有类别的概率
        all_probs = probabilities[0].cpu().numpy()
        prob_dict = {self.labels[i]: float(all_probs[i]) for i in range(len(self.labels))}
        
        result = {
            'text': text,
            'predicted_label': self.idx2label[predicted_idx],
            'confidence': confidence,
            'all_probabilities': prob_dict
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        批量预测
        
        参数:
            texts: 文本列表
        
        返回:
            预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_topk(self, text, k=3):
        """
        预测Top-K类别
        
        参数:
            text: 输入文本
            k: 返回前k个最可能的类别
        
        返回:
            Top-K预测结果
        """
        # 文本转序列
        sequence = self.text_processor.text_to_sequence(text)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # 获取Top-K
        topk_probs, topk_indices = torch.topk(probabilities, k)
        
        results = []
        for i in range(k):
            idx = topk_indices[i].item()
            prob = topk_probs[i].item()
            results.append({
                'label': self.idx2label[idx],
                'probability': prob
            })
        
        return {
            'text': text,
            'top_k_predictions': results
        }
    
    def print_prediction(self, result):
        """
        打印预测结果
        
        参数:
            result: predict方法返回的结果字典
        """
        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"\n输入文本: {result['text']}")
        print(f"\n预测类别: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("\n各类别概率:")
        print("-" * 40)
        
        # 按概率排序
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            bar = "█" * int(prob * 30)
            print(f"  {label:<10} {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        print("=" * 60)


def load_predictor(model_path, vocab_path, model_class, config, device='cpu'):
    """
    加载预测器
    
    参数:
        model_path: 模型权重路径
        vocab_path: 词表路径
        model_class: 模型类
        config: 配置字典
        device: 计算设备
    
    返回:
        Predictor实例
    """
    from utils.data_processor import TextProcessor
    
    # 加载词表
    text_processor = TextProcessor()
    text_processor.load_vocab(vocab_path)
    
    # 创建模型
    vocab_size = len(text_processor.word2idx)
    num_classes = config['num_classes']
    model = model_class(vocab_size, num_classes, config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载标签映射
    label2idx = config.get('label2idx', {})
    
    # 创建预测器
    predictor = Predictor(model, text_processor, label2idx, device)
    
    return predictor
