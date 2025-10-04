import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np

from model import CIFAR10Net, count_parameters
from transforms import get_train_transforms, get_test_transforms, AlbumentationTransform


def test_model():
    """Test model architecture and forward pass"""
    print("=== Testing Model Architecture ===")
    
    model = CIFAR10Net()
    total_params = count_parameters(model)
    rf = model.calculate_receptive_field()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Receptive field: {rf}")
    print(f"Parameters < 200k: {total_params < 200000}")
    print(f"RF > 44: {rf > 44}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful ✅")
    
    return total_params < 200000 and rf > 44


def test_transforms():
    """Test albumentation transforms"""
    print("\n=== Testing Transforms ===")
    
    # Test transforms
    train_transform = AlbumentationTransform(get_train_transforms())
    test_transform = AlbumentationTransform(get_test_transforms())
    
    # Create dummy CIFAR-10 image
    dummy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Test train transform
    train_tensor = train_transform(dummy_img)
    print(f"Train transform output shape: {train_tensor.shape}")
    print(f"Train transform output type: {type(train_tensor)}")
    
    # Test test transform
    test_tensor = test_transform(dummy_img)
    print(f"Test transform output shape: {test_tensor.shape}")
    print(f"Test transform output type: {type(test_tensor)}")
    
    print("Transforms working correctly ✅")
    return True


def test_dataset_loading():
    """Test dataset loading with transforms"""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        # Test loading a small batch
        train_transform = AlbumentationTransform(get_train_transforms())
        
        # Just test if we can create the dataset object
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=False, transform=train_transform
        )
        
        print(f"Dataset length: {len(train_dataset)}")
        print("Dataset created successfully ✅")
        return True
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("This is expected if CIFAR-10 hasn't been downloaded yet")
        return False


def main():
    """Run all tests"""
    print("CIFAR-10 CNN Setup Verification")
    print("=" * 40)
    
    # Test model
    model_ok = test_model()
    
    # Test transforms
    transforms_ok = test_transforms()
    
    # Test dataset (might fail if not downloaded)
    dataset_ok = test_dataset_loading()
    
    print("\n=== Summary ===")
    print(f"Model architecture: {'✅' if model_ok else '❌'}")
    print(f"Transforms: {'✅' if transforms_ok else '❌'}")
    print(f"Dataset loading: {'✅' if dataset_ok else '⚠️ (download needed)'}")
    
    if model_ok and transforms_ok:
        print("\n🎉 Setup verified! Ready to train.")
        print("Run 'python train.py' to start training.")
    else:
        print("\n❌ Setup issues detected. Please check the errors above.")


if __name__ == "__main__":
    main()