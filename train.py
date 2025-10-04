import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime
import os
from torchsummary import summary

from model import CIFAR10Net, count_parameters
from transforms import get_train_transforms, get_test_transforms, AlbumentationTransform


def setup_logging():
    """Setup logging to save training logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_filepath = os.path.join("logs", log_filename)
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_filepath


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets():
    """Load CIFAR-10 datasets with transforms"""
    train_transform = AlbumentationTransform(get_train_transforms())
    test_transform = AlbumentationTransform(get_test_transforms())
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    return train_dataset, test_dataset


def get_data_loaders(batch_size=128):
    """Create data loaders"""
    train_dataset, test_dataset = get_datasets()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Log training results
    logging.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Epoch {epoch} - Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Log validation results
    logging.info(f"Epoch {epoch} Validation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    return test_loss, test_acc


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Setup logging
    log_filepath = setup_logging()
    logging.info("=== CIFAR-10 Training Started ===")
    logging.info(f"Logs will be saved to: {log_filepath}")
    
    # Set random seed
    set_seed(42)
    logging.info("Random seed set to 42")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create model
    model = CIFAR10Net().to(device)
    total_params = count_parameters(model)
    rf = model.calculate_receptive_field()
    
    logging.info('=== Model Architecture ===')
    logging.info(f'Total parameters: {total_params:,}')
    logging.info(f'Receptive field: {rf}')
    logging.info(f'Parameters < 200k: {total_params < 200000}')
    logging.info(f'RF > 44: {rf > 44}')
    
    # Print torch summary
    logging.info('=== Model Summary ===')
    try:
        summary_str = str(summary(model, (3, 32, 32), verbose=0))
        logging.info(f"Torch Summary:\n{summary_str}")
    except Exception as e:
        logging.warning(f"Could not generate torch summary: {e}")
    
    if total_params >= 200000:
        logging.error(f'WARNING: Model has {total_params} parameters, exceeding 200k limit!')
        return
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(batch_size=128)
    logging.info(f'Data loaders created - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=100, steps_per_epoch=len(train_loader),
        pct_start=0.1, div_factor=10, final_div_factor=100
    )
    
    logging.info('=== Training Configuration ===')
    logging.info(f'Loss function: CrossEntropyLoss')
    logging.info(f'Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)')
    logging.info(f'Scheduler: OneCycleLR (max_lr=0.1)')
    logging.info(f'Batch size: 128')
    
    # Training loop
    num_epochs = 100
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0.0
    
    logging.info(f'\n=== Starting Training for {num_epochs} epochs ===')
    
    for epoch in range(1, num_epochs + 1):
        logging.info(f'\n--- Epoch [{epoch}/{num_epochs}] ---')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate after each epoch
        val_loss, val_acc = validate(model, test_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log epoch summary
        logging.info(f'Epoch {epoch} Summary:')
        logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logging.info(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }, 'best_model.pth')
            logging.info(f'  *** New best model saved with accuracy: {best_acc:.2f}% ***')
        
        # Early stopping if target accuracy reached
        if val_acc >= 85.0:
            logging.info(f'\n🎉 Target accuracy of 85% reached! Best accuracy: {best_acc:.2f}%')
            break
    
    logging.info(f'\n=== Training Completed ===')
    logging.info(f'Best validation accuracy: {best_acc:.2f}%')
    logging.info(f'Target achieved (>=85%): {best_acc >= 85.0}')
    
    # Save final training log summary
    logging.info('\n=== Final Training Summary ===')
    logging.info(f'Total epochs completed: {epoch}')
    logging.info(f'Final train accuracy: {train_accs[-1]:.2f}%')
    logging.info(f'Final validation accuracy: {val_accs[-1]:.2f}%')
    logging.info(f'Best validation accuracy: {best_acc:.2f}%')
    
    # Plot metrics
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    
    return model, best_acc


if __name__ == '__main__':
    model, best_acc = main()