#!/usr/bin/env python
import os
import sys
import subprocess

def run_command(cmd, cwd=None):
    print(f"\n执行: {cmd}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"命令执行失败: {cmd}")
        sys.exit(1)
    return result

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("中文文本分类项目 - 完整流程")
    print("=" * 60)
    
    print("\n步骤 1/4: 生成数据集")
    run_command("python data/generate_data.py", cwd=base_dir)
    
    print("\n步骤 2/4: 训练模型")
    run_command("python train.py --num_epochs 15", cwd=base_dir)
    
    print("\n步骤 3/4: 评估模型")
    run_command("python evaluate.py", cwd=base_dir)
    
    print("\n步骤 4/4: 预测示例")
    run_command("python predict.py", cwd=base_dir)
    
    print("\n" + "=" * 60)
    print("所有步骤完成！")
    print("=" * 60)
    print("\n使用方法:")
    print("  - 训练模型: python train.py")
    print("  - 评估模型: python evaluate.py")
    print("  - 预测文本: python predict.py --text '你的文本'")
    print("  - 交互预测: python predict.py --interactive")

if __name__ == "__main__":
    main()
