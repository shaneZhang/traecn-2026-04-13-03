# -*- coding: utf-8 -*-
"""
模型评估模块
包括：准确率、混淆矩阵、精确率、召回率、F1分数
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class Evaluator:
    """模型评估器"""
    
    def __init__(self, label_names):
        """
        参数:
            label_names: 标签名称列表
        """
        self.label_names = label_names
        self.num_classes = len(label_names)
    
    def calculate_metrics(self, y_true, y_pred):
        """
        计算各项评估指标
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
        
        返回:
            包含各项指标的字典
        """
        metrics = {}
        
        # 准确率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 精确率（每个类别和宏平均）
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        # 召回率（每个类别和宏平均）
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # F1分数（每个类别和宏平均）
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        
        # 整体准确率
        print(f"\n【整体准确率】")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        # 宏平均指标
        print(f"\n【宏平均指标 (Macro Average)】")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
        
        # 加权平均指标
        print(f"\n【加权平均指标 (Weighted Average)】")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
        
        # 每个类别的指标
        print(f"\n【各类别详细指标】")
        print("-" * 60)
        print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        
        for i, label in enumerate(self.label_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            print(f"{label:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        print("=" * 60)
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        绘制混淆矩阵热力图
        
        参数:
            metrics: 包含混淆矩阵的字典
            save_path: 保存路径（可选）
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n混淆矩阵已保存至: {save_path}")
        
        plt.show()
    
    def evaluate(self, y_true, y_pred, plot_cm=True, save_cm_path=None):
        """
        完整评估流程
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            plot_cm: 是否绘制混淆矩阵
            save_cm_path: 混淆矩阵保存路径
        
        返回:
            评估指标字典
        """
        # 计算指标
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 打印指标
        self.print_metrics(metrics)
        
        # 绘制混淆矩阵
        if plot_cm:
            self.plot_confusion_matrix(metrics, save_cm_path)
        
        return metrics
    
    def get_classification_report(self, y_true, y_pred, digits=4):
        """
        获取sklearn的分类报告
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            digits: 小数位数
        
        返回:
            分类报告字符串
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.label_names,
            digits=digits
        )
        return report


def evaluate_model(model, test_loader, device, label_names, plot_cm=True, save_cm_path=None):
    """
    便捷函数：评估模型
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        label_names: 标签名称列表
        plot_cm: 是否绘制混淆矩阵
        save_cm_path: 混淆矩阵保存路径
    
    返回:
        评估指标字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    import torch
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 创建评估器并评估
    evaluator = Evaluator(label_names)
    metrics = evaluator.evaluate(all_labels, all_preds, plot_cm=plot_cm, save_cm_path=save_cm_path)
    
    return metrics
