"""
Optimizer and Learning Rate Scheduler Configuration
"""

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_optimizer(model, config):
    """
    Build optimizer based on config

    Args:
        model: PyTorch model
        config: dict with optimizer configuration
            - optimizer: 'adam' or 'adamw' or 'sgd'
            - lr: learning rate
            - weight_decay: L2 regularization
            - momentum: for SGD
            - betas: for Adam/AdamW

    Returns:
        optimizer: configured optimizer
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)

    # Separate parameters for different learning rates
    # Freeze LLM parameters (already frozen in model)
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999))
        )

    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999))
        )

    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=True
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"✅ Optimizer: {optimizer_name.upper()}")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Trainable parameters: {sum(p.numel() for p in params):,}")

    return optimizer


def build_scheduler(optimizer, config, num_epochs):
    """
    Build learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        config: dict with scheduler configuration
            - scheduler: 'cosine' or 'plateau' or 'step'
            - warmup_epochs: number of warmup epochs
        num_epochs: total number of training epochs

    Returns:
        scheduler: configured scheduler
    """
    scheduler_name = config.get('scheduler', 'cosine').lower()
    warmup_epochs = config.get('warmup_epochs', 0)

    if scheduler_name == 'cosine':
        # Cosine annealing with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )

    elif scheduler_name == 'plateau':
        # Reduce on plateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize mAP
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5),
            min_lr=config.get('min_lr', 1e-6),
            verbose=True
        )

    elif scheduler_name == 'step':
        # Step decay
        scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.5)
        )

    else:
        scheduler = None
        print(f"⚠️  No scheduler: using constant learning rate")

    if scheduler is not None:
        print(f"✅ Scheduler: {scheduler_name.upper()}")
        if warmup_epochs > 0:
            print(f"   Warmup epochs: {warmup_epochs}")

    return scheduler


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    Gradually increases learning rate from 0 to base_lr over warmup_epochs
    """

    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.current_epoch += 1

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Optimizer Configuration")
    print("=" * 60)

    import torch.nn as nn


    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 6)

        def forward(self, x):
            return self.fc2(self.fc1(x))


    model = DummyModel()

    # Test optimizer configuration
    print("\n📊 Testing optimizer configuration...")
    config = {
        'optimizer': 'adamw',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999)
    }

    optimizer = build_optimizer(model, config)
    print(f"   Optimizer type: {type(optimizer).__name__}")

    # Test scheduler configuration
    print("\n📊 Testing scheduler configuration...")
    scheduler_config = {
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'min_lr': 1e-6
    }

    scheduler = build_scheduler(optimizer, scheduler_config, num_epochs=50)
    print(f"   Scheduler type: {type(scheduler).__name__}")

    # Test warmup scheduler
    print("\n📊 Testing warmup scheduler...")
    warmup = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=1e-4)

    print("\n   Warmup learning rates:")
    for epoch in range(7):
        lr = warmup.get_lr()
        print(f"      Epoch {epoch}: lr = {lr:.6f}")
        warmup.step()

    print("\n" + "=" * 60)
    print("✅ Optimizer Configuration Test PASSED!")
    print("=" * 60)