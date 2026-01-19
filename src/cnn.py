import torch
import torch.nn as nn
import numpy as np

class NatureCNN(nn.Module):
    """
    CNN architecture described in 'Human-level control through deep reinforcement learning' (Mnih et al., 2015).
    This is the standard 'backbone' for many Atari and Genesis RL agents.
    
    Expects input shape of (Channels, Height, Width), typically (4, 84, 84).
    """
    def __init__(self, input_shape, features_dim=512):
        super().__init__()
        n_input_channels = input_shape[0] # Number of layers (stacked frames)
        
        # --- THE CONVOLUTIONAL BACKBONE ---
        # These layers act like filters that scan the screen for visual patterns.
        self.cnn = nn.Sequential(
            # Layer 1: Sees broad shapes (8x8 pixel patterns).
            # Output shape: (Batch_size, 32 channels, 20 height, 20 width)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: Sees smaller, more specific patterns (4x4).
            # Output shape: (Batch_size, 64 channels, 9 height, 9 width)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: Sees fine-grained details (3x3).
            # Output shape: (Batch_size, 64 channels, 7 height, 7 width)
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            
            # Layer 4: Squash the 2D grid into a 1D list of numbers (Flattening).
            # Output shape: (Batch_size, 64 * 7 * 7 = 3136 features)
            nn.Flatten(),
        )

        # We calculate the number of features after flattening so the next layer knows its input size.
        with torch.no_grad():
            dummy_input = torch.as_tensor(np.zeros((1, *input_shape))).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        # --- THE FEATURE BED ---
        # A fully connected layer that condenses the 3136 raw features into 512 high-level "concepts."
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with observation normalization.
        Genesis pixels are [0, 255]. We divide by 255.0 to get [0, 1].
        Neural networks learn much faster when numbers are small and normalized.
        """
        return self.linear(self.cnn(observations / 255.0))
