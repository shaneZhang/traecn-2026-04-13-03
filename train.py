import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

from config import Config
from utils.data_utils import load_data
from utils.eval_utils import calculate_metrics, print_metrics, plot_confusion_matrix
from models.textcnn import TextCNN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics

def evaluate(model, loader, criterion, device, desc='Evaluating'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc=desc):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_preds, all_labels

def main():
    config = Config()
    set_seed(config.seed)
    
    print(f'Using device: {config.device}')
    print('=' * 60)
    
    train_loader, val_loader, test_loader, embedding_matrix, vocab = load_data()
    print('=' * 60)
    
    model = TextCNN(embedding_matrix).to(config.device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    
    best_val_acc = 0
    
    print('=' * 60)
    print('Start training...')
    print('=' * 60)
    
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)
        
        print(f'\nTrain Loss: {train_loss:.4f}, Acc: {train_metrics["accuracy"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_metrics["accuracy"]:.4f}')
        
        scheduler.step(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), config.save_path)
            print(f'Best model saved! Val Acc: {best_val_acc:.4f}')
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print('=' * 60)
    
    print('\nLoading best model for testing...')
    model.load_state_dict(torch.load(config.save_path))
    
    test_loss, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device, desc='Testing')
    print('\n' + '=' * 60)
    print('Test Set Results:')
    print('=' * 60)
    print_metrics(test_metrics)
    
    plot_confusion_matrix(test_labels, test_preds)
    
    print('\n' + '=' * 60)
    print('All done!')
    print('=' * 60)

if __name__ == '__main__':
    main()
