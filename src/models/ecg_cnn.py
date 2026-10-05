"""
CardioSense - Lightweight 1D-CNN Waveform Architecture
Deep convolutional neural network processing raw 12-lead ECG time-series directly.
Parameter-efficient (~45,000 parameters), optimized for CPU real-time inference.
"""

import torch
import torch.nn as nn


class CardioNet1D(nn.Module):
    """
    1D Convolutional Neural Network for multi-lead ECG classification.
    Processes (Batch, 12 Channels, 1000 Timepoints).
    """

    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()

        # Block 1: Broad receptive field (kernel 7) to capture overall beat shape
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 1000 -> 500 samples
        )

        # Block 2: Medium receptive field (kernel 5) for QRS and ST repolarization
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 500 -> 250 samples
        )

        # Block 3: High-level morphological feature maps
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # Global Average Pooling -> 128 embedding vector
        )

        # Classifier Head with Dropout for regularized generalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.35),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Forward pass through convolutional blocks and classifier."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        logits = self.classifier(x)
        return logits

    def extract_embedding(self, x):
        """Extracts the 128-dimensional latent vector for ensemble fusion."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        embedding = torch.flatten(x, 1)
        return embedding