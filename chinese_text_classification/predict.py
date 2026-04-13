# -*- coding: utf-8 -*-
"""
预测脚本
使用训练好的模型对新文本进行分类
"""

import os
import sys
import torch
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.textcnn import TextCNN
from utils.data_processor import TextProcessor
from utils.predictor import Predictor


def load_model_and_processor(model_path, vocab_path, config):
    """加载模型和文本处理器"""
    # 加载词表
    text_processor = TextProcessor()
    text_processor.load_vocab(vocab_path)
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    label2idx = checkpoint['label2idx']
    
    # 创建模型
    vocab_size = len(text_processor.word2idx)
    num_classes = len(label2idx)
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        num_classes=num_classes,
        num_filters=config['model']['num_filters'],
        filter_sizes=config['model']['filter_sizes'],
        dropout=config['model']['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, text_processor, label2idx


def interactive_predict(predictor):
    """交互式预测"""
    print("\n" + "=" * 60)
    print("交互式预测模式")
    print("=" * 60)
    print("请输入中文文本进行分类预测（输入'quit'退出）:\n")
    
    while True:
        try:
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n感谢使用，再见!")
                break
            
            if not text:
                continue
            
            # 预测
            result = predictor.predict(text)
            predictor.print_prediction(result)
            
        except KeyboardInterrupt:
            print("\n\n感谢使用，再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue


def batch_predict(predictor, texts):
    """批量预测"""
    print("\n" + "=" * 60)
    print("批量预测结果")
    print("=" * 60)
    
    results = predictor.predict_batch(texts)
    
    for i, result in enumerate(results, 1):
        print(f"\n【样本 {i}】")
        print(f"文本: {result['text']}")
        print(f"预测类别: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中文文本分类预测')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='模型路径')
    parser.add_argument('--vocab', type=str, default='models/vocab.pkl',
                        help='词表路径')
    parser.add_argument('--text', type=str, default=None,
                        help='待预测的文本')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式预测模式')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("中文文本分类 - 预测")
    print("=" * 60)
    
    # 加载配置
    config = get_config()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"\n错误: 模型文件不存在: {args.model}")
        print("请先运行训练脚本: python train.py")
        return
    
    if not os.path.exists(args.vocab):
        print(f"\n错误: 词表文件不存在: {args.vocab}")
        print("请先运行训练脚本: python train.py")
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model, text_processor, label2idx = load_model_and_processor(
        args.model, args.vocab, config
    )
    
    # 创建预测器
    device = torch.device('cpu')
    predictor = Predictor(model, text_processor, label2idx, device)
    
    print(f"类别: {list(label2idx.keys())}")
    
    # 根据参数选择预测模式
    if args.interactive:
        interactive_predict(predictor)
    elif args.text:
        result = predictor.predict(args.text)
        predictor.print_prediction(result)
    else:
        # 默认示例预测
        print("\n" + "-" * 60)
        print("示例预测（使用默认文本）")
        print("-" * 60)
        
        sample_texts = [
            "政府召开重要会议，讨论经济发展相关政策，强调改革的重要性。",
            "人工智能技术取得重大突破，在图像识别领域迎来新发展。",
            "中国足球队在世界杯预选赛中表现出色，获得关键胜利。",
            "著名演员主演的新电影即将上映，票房备受期待。",
            "股市今日表现强劲，科技股板块领涨，创下新高。"
        ]
        
        for i, text in enumerate(sample_texts):
            result = predictor.predict(text)
            predictor.print_prediction(result)
            if i < len(sample_texts) - 1:
                try:
                    input("\n按Enter键继续...")
                except EOFError:
                    pass
        
        # 进入交互模式
        print("\n" + "-" * 60)
        try:
            response = input("是否进入交互式预测模式? (y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                interactive_predict(predictor)
        except EOFError:
            pass


if __name__ == "__main__":
    main()
