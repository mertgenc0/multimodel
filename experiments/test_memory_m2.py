"""
Memory Test for M2 MacBook Air
Tests if model fits in memory before training
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.baseline.baseline_model import BaselineModel


def test_memory():
    print("🧪 Testing Memory Requirements on M2...")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    print("\n🏗️  Creating model...")
    model = BaselineModel(
        num_classes=6,
        hidden_dim=256,
        pretrained_resnet=True
    ).to(device)

    print("✅ Model created successfully!")

    # Test forward pass with different batch sizes
    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        print(f"\n📊 Testing batch size {bs}...")

        try:
            # Create dummy data
            images = torch.randn(bs, 3, 224, 224).to(device)
            captions = [f"Test caption {i}" for i in range(bs)]

            # Forward pass
            with torch.no_grad():
                outputs = model(images, captions)

            print(f"   ✅ Batch size {bs} works!")
            print(f"      Output shape: {outputs['logits'].shape}")

        except Exception as e:
            print(f"   ❌ Batch size {bs} failed: {e}")
            break

    print("\n🎉 Memory test complete!")


if __name__ == "__main__":
    test_memory()