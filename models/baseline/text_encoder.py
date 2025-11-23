"""
Text Encoder Module for Baseline Model
Uses LLM (BERT) + CNN to extract text features
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class LLMTextEncoder(nn.Module):
    """
    LLM + CNN based text encoder

    Pipeline:
    1. BERT tokenizes and encodes text → [batch, seq_len, 768]
    2. CNN extracts local features → [batch, hidden_dim]
    3. FC layer projects to final dimension
    """

    def __init__(self, model_name='bert-base-uncased', hidden_dim=256):
        super(LLMTextEncoder, self).__init__()

        print(f"🔧 Initializing Text Encoder with {model_name}...")

        # LLM (BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModel.from_pretrained(model_name)

        # Freeze LLM parameters (save memory & training time)
        for param in self.llm.parameters():
            param.requires_grad = False

        llm_dim = self.llm.config.hidden_size  # 768 for BERT-base

        # Text CNN (from baseline paper)
        # Kernel size = 3, stride = 1 (as per Algorithm 1)
        self.text_cnn = nn.Sequential(
            nn.Conv1d(llm_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Final projection
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        print(f"✅ Text Encoder initialized")
        print(f"   LLM dimension: {llm_dim}")
        print(f"   Output dimension: {hidden_dim}")

    def forward(self, captions):
        """
        Args:
            captions: List[str] - batch of text captions

        Returns:
            text_features: Tensor [batch_size, hidden_dim]
        """
        # Tokenize captions
        device = next(self.llm.parameters()).device

        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=77,  # Standard for CLIP-like models
            return_tensors='pt'
        ).to(device)

        # LLM encoding (frozen, no gradients)
        with torch.no_grad():
            outputs = self.llm(**tokens)
            # Get last hidden state: [batch, seq_len, 768]
            text_embeddings = outputs.last_hidden_state

        # Transpose for CNN: [batch, 768, seq_len]
        text_embeddings = text_embeddings.permute(0, 2, 1)

        # CNN feature extraction
        features = self.text_cnn(text_embeddings)  # [batch, hidden_dim, 1]
        features = features.squeeze(-1)  # [batch, hidden_dim]

        # Final projection
        text_features = self.fc(features)

        return text_features

    def get_output_dim(self):
        """Return output feature dimension"""
        return self.fc.out_features


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Text Encoder Module")
    print("=" * 60)

    # Create model
    model = LLMTextEncoder(hidden_dim=256)
    model.eval()

    # Test with sample captions
    captions = [
        "Two friends are playing basketball on the court",
        "A couple walking hand in hand in the park",
        "Family members having dinner together",
        "Colleagues working in an office"
    ]

    print(f"\n📝 Test captions ({len(captions)} samples):")
    for i, cap in enumerate(captions):
        print(f"   {i + 1}. {cap}")

    # Forward pass
    print(f"\n⚙️  Running forward pass...")
    with torch.no_grad():
        features = model(captions)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input: {len(captions)} captions")
    print(f"   Output shape: {features.shape}")  # Should be [4, 256]
    print(f"   Output dtype: {features.dtype}")
    print(f"   Output range: [{features.min():.3f}, {features.max():.3f}]")

    # Test with single caption
    print(f"\n🔍 Testing single caption...")
    single_caption = ["A young couple smiling together"]
    with torch.no_grad():
        single_feature = model(single_caption)

    print(f"   Single output shape: {single_feature.shape}")  # Should be [1, 256]

    # Memory usage
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {param_count:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {param_count - trainable_params:,}")

    print("\n" + "=" * 60)
    print("✅ Text Encoder Test PASSED!")
    print("=" * 60)