"""
Multimodal Fusion Module for Baseline Model
Uses MLP to learn adaptive weights for fusing image and text features
"""

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """
    Learns to adaptively fuse image and text features

    Pipeline:
    1. Concatenate aligned image and text features
    2. MLP learns fusion weight w ∈ [0, 1]
    3. Fused feature = w * image + (1-w) * text

    From baseline paper Equation (3):
    F_fusion = w ⊙ F_I + (1 - w) ⊙ F_T
    where w = MLP([F_I; F_T])
    """

    def __init__(self, feature_dim=256, hidden_dim=128):
        super(AdaptiveFusion, self).__init__()

        print(f"🔧 Initializing Adaptive Fusion Module...")

        # MLP to learn fusion weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()  # Output weights in [0, 1]
        )

        # Optional: Additional transformation after fusion
        self.fusion_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        print(f"✅ Adaptive Fusion initialized")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Hidden dimension: {hidden_dim}")

    def forward(self, image_features, text_features):
        """
        Adaptively fuse image and text features

        Args:
            image_features: [batch_size, feature_dim] - aligned image features
            text_features: [batch_size, feature_dim] - aligned text features

        Returns:
            fused_features: [batch_size, feature_dim]
            weights: [batch_size, feature_dim] - fusion weights (for analysis)
        """
        # Concatenate features
        concat_features = torch.cat([image_features, text_features], dim=-1)
        # [batch, feature_dim * 2]

        # Learn fusion weights via MLP
        weights = self.weight_mlp(concat_features)  # [batch, feature_dim]

        # Adaptive weighted fusion
        fused = weights * image_features + (1 - weights) * text_features
        # [batch, feature_dim]

        # Optional transformation
        fused_features = self.fusion_transform(fused)

        return fused_features, weights


class SimpleFusion(nn.Module):
    """
    Simple baseline fusion: concatenate and project
    Used for ablation studies
    """

    def __init__(self, feature_dim=256):
        super(SimpleFusion, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, image_features, text_features):
        """Simple concatenation and projection"""
        concat = torch.cat([image_features, text_features], dim=-1)
        fused = self.fusion(concat)
        return fused, None  # No weights for simple fusion


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Fusion Module")
    print("=" * 60)

    # Create fusion modules
    print("\n📊 Testing Adaptive Fusion...")
    adaptive_fusion = AdaptiveFusion(feature_dim=256, hidden_dim=128)
    adaptive_fusion.eval()

    print("\n📊 Testing Simple Fusion...")
    simple_fusion = SimpleFusion(feature_dim=256)
    simple_fusion.eval()

    # Test with sample features
    print(f"\n📊 Creating test features...")
    batch_size = 4
    feature_dim = 256

    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)

    print(f"   Image features shape: {image_features.shape}")
    print(f"   Text features shape: {text_features.shape}")

    # Test Adaptive Fusion
    print(f"\n⚙️  Running adaptive fusion...")
    with torch.no_grad():
        fused_adaptive, weights = adaptive_fusion(image_features, text_features)

    print(f"\n✅ Adaptive fusion successful!")
    print(f"   Fused features shape: {fused_adaptive.shape}")  # [4, 256]
    print(f"   Fusion weights shape: {weights.shape}")  # [4, 256]
    print(f"   Weight statistics:")
    print(f"     Mean: {weights.mean():.3f}")
    print(f"     Std: {weights.std():.3f}")
    print(f"     Min: {weights.min():.3f}, Max: {weights.max():.3f}")
    print(f"   Interpretation: {weights.mean():.1%} image, {(1 - weights.mean()):.1%} text")

    # Test Simple Fusion
    print(f"\n⚙️  Running simple fusion...")
    with torch.no_grad():
        fused_simple, _ = simple_fusion(image_features, text_features)

    print(f"\n✅ Simple fusion successful!")
    print(f"   Fused features shape: {fused_simple.shape}")  # [4, 256]

    # Compare outputs
    print(f"\n🔍 Comparing fusion methods...")
    print(f"   Adaptive fusion output range: [{fused_adaptive.min():.3f}, {fused_adaptive.max():.3f}]")
    print(f"   Simple fusion output range: [{fused_simple.min():.3f}, {fused_simple.max():.3f}]")

    # Test edge cases
    print(f"\n🧪 Testing edge cases...")

    # Case 1: Identical features (should give equal weights ~0.5)
    identical = torch.randn(2, feature_dim)
    with torch.no_grad():
        _, identical_weights = adaptive_fusion(identical, identical)
    print(f"   Identical features → weights: {identical_weights.mean():.3f} (should be ~0.5)")

    # Case 2: Very different features
    strong_image = torch.randn(2, feature_dim) * 10  # Strong signal
    weak_text = torch.randn(2, feature_dim) * 0.1  # Weak signal
    with torch.no_grad():
        _, diff_weights = adaptive_fusion(strong_image, weak_text)
    print(f"   Strong image + weak text → weights: {diff_weights.mean():.3f}")
    print(f"   (Higher weight = more reliance on image)")

    # Test single sample
    print(f"\n🔍 Testing single sample...")
    single_image = torch.randn(1, feature_dim)
    single_text = torch.randn(1, feature_dim)
    with torch.no_grad():
        single_fused, single_weights = adaptive_fusion(single_image, single_text)

    print(f"   Single fused shape: {single_fused.shape}")  # [1, 256]
    print(f"   Single weights mean: {single_weights.mean():.3f}")

    # Memory usage
    adaptive_params = sum(p.numel() for p in adaptive_fusion.parameters())
    adaptive_trainable = sum(p.numel() for p in adaptive_fusion.parameters() if p.requires_grad)

    simple_params = sum(p.numel() for p in simple_fusion.parameters())
    simple_trainable = sum(p.numel() for p in simple_fusion.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Adaptive Fusion:")
    print(f"     Total parameters: {adaptive_params:,}")
    print(f"     Trainable parameters: {adaptive_trainable:,}")
    print(f"   Simple Fusion:")
    print(f"     Total parameters: {simple_params:,}")
    print(f"     Trainable parameters: {simple_trainable:,}")

    print("\n" + "=" * 60)
    print("✅ Fusion Module Test PASSED!")
    print("=" * 60)

    print("\n💡 Practical Interpretation:")
    print("   - Adaptive fusion learns to weight modalities dynamically")
    print("   - Weight ~0.5 → both modalities equally important")
    print("   - Weight >0.5 → rely more on image")
    print("   - Weight <0.5 → rely more on text")
    print("   - Simple fusion is faster but less flexible")