"""
Convolutional Energy-Based Model implementation.

Lightweight architecture optimized for Kaggle with CIFAR-10.
Uses ResNet-style blocks for efficient energy computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    Residual block for ConvNet energy model.
    
    Optimized for Kaggle: Smaller channel dimensions.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
            downsample: Whether to downsample
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = Swish()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out


class ConvEBM(nn.Module):
    """
    Convolutional Energy-Based Model.
    
    Small ResNet architecture that outputs scalar energy.
    Optimized for CIFAR-10 on Kaggle: ~4M parameters.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,  # Reduced from 128 for Kaggle
        num_blocks: list = [2, 2, 2],  # Reduced from [2, 2, 2, 2]
        spectral_norm: bool = True
    ):
        """
        Initialize ConvEBM.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (kept small for Kaggle)
            num_blocks: Number of residual blocks per stage
            spectral_norm: Whether to use spectral normalization
        """
        super(ConvEBM, self).__init__()
        
        self.input_channels = input_channels
        self.spectral_norm = spectral_norm
        
        # Initial convolution
        self.conv1 = nn.Conv2d(
            input_channels, base_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.activation = Swish()
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, num_blocks[2], stride=2)
        
        # Energy head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, 1)
        
        # Apply spectral normalization if requested
        if spectral_norm:
            self._apply_spectral_norm()
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to all conv and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.utils.spectral_norm(module)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy E(x) for input samples.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Energy values [batch_size]
        """
        # Initial conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global pooling and energy output
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        energy = self.fc(out).squeeze()
        
        return energy
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before energy head."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        return out


class TinyConvEBM(nn.Module):
    """
    Tiny ConvEBM for very fast experiments.
    
    ~1M parameters, suitable for quick prototyping on Kaggle.
    """
    
    def __init__(self, input_channels: int = 3):
        super(TinyConvEBM, self).__init__()
        
        self.net = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            Swish(),
            nn.Conv2d(32, 32, 4, 2, 1),
            Swish(),
            
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, 1, 1),
            Swish(),
            nn.Conv2d(64, 64, 4, 2, 1),
            Swish(),
            
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, 1, 1),
            Swish(),
            nn.Conv2d(128, 128, 4, 2, 1),
            Swish(),
            
            # 4x4 -> 1
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(128, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        features = features.view(features.size(0), -1)
        energy = self.fc(features).squeeze()
        return energy


def build_conv_ebm(
    model_size: str = 'small',
    input_channels: int = 3,
    spectral_norm: bool = True
) -> nn.Module:
    """
    Build ConvEBM model.
    
    Args:
        model_size: 'tiny' (~1M params), 'small' (~4M params), 'medium' (~10M params)
        input_channels: Number of input channels
        spectral_norm: Use spectral normalization
        
    Returns:
        ConvEBM model
    """
    if model_size == 'tiny':
        return TinyConvEBM(input_channels=input_channels)
    elif model_size == 'small':
        return ConvEBM(
            input_channels=input_channels,
            base_channels=64,
            num_blocks=[2, 2, 2],
            spectral_norm=spectral_norm
        )
    elif model_size == 'medium':
        return ConvEBM(
            input_channels=input_channels,
            base_channels=96,
            num_blocks=[2, 2, 2, 2],
            spectral_norm=spectral_norm
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")


if __name__ == "__main__":
    print("Testing ConvEBM models...")
    
    # Test input
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Test tiny model
    print("\n1. Tiny ConvEBM:")
    tiny_model = build_conv_ebm('tiny')
    energy = tiny_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Energy shape: {energy.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tiny_model.parameters()):,}")
    print(f"   ✓ Tiny model works!")
    
    # Test small model
    print("\n2. Small ConvEBM:")
    small_model = build_conv_ebm('small')
    energy = small_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Energy shape: {energy.shape}")
    print(f"   Parameters: {sum(p.numel() for p in small_model.parameters()):,}")
    print(f"   ✓ Small model works!")
    
    # Test medium model
    print("\n3. Medium ConvEBM:")
    medium_model = build_conv_ebm('medium')
    energy = medium_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Energy shape: {energy.shape}")
    print(f"   Parameters: {sum(p.numel() for p in medium_model.parameters()):,}")
    print(f"   ✓ Medium model works!")
    
    # Test gradient computation
    print("\n4. Testing gradient computation:")
    x.requires_grad = True
    energy = small_model(x)
    grad = torch.autograd.grad(energy.sum(), x)[0]
    print(f"   Gradient shape: {grad.shape}")
    print(f"   Gradient norm: {grad.norm().item():.4f}")
    print(f"   ✓ Gradient computation works!")
    
    # Memory estimate
    print("\n5. Memory estimates (batch_size=64):")
    x_large = torch.randn(64, 3, 32, 32)
    
    for name, model in [('Tiny', tiny_model), ('Small', small_model), ('Medium', medium_model)]:
        params_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
        print(f"   {name}: ~{params_mb:.1f}MB parameters")
    
    print("\n✅ All ConvEBM tests passed!")