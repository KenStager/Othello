"""
Cross-platform neural network inference optimization.

Automatically applies appropriate optimizations based on device type:
- MPS (Apple Silicon): FP16 + Channels Last + TorchScript
- CUDA (NVIDIA): FP16 + Channels Last + TensorRT
- CPU: Channels Last only (FP16 may be slower on CPU)
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """
    Optimizes neural network models for inference based on target device.

    Supports:
    - Apple Silicon (MPS): FP16, Channels Last, TorchScript
    - NVIDIA GPUs (CUDA): FP16, Channels Last, TensorRT
    - CPU: Channels Last only
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimizer with configuration.

        Args:
            config: Optimization config dict with keys:
                - precision: "fp16" or "fp32" (default: "fp16")
                - use_channels_last: bool (default: True)
                - use_compilation: bool (default: True)
                - tensorrt_workspace_gb: int (default: 1)
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 'fp16')
        self.use_channels_last = self.config.get('use_channels_last', True)
        self.use_compilation = self.config.get('use_compilation', True)
        self.tensorrt_workspace_gb = self.config.get('tensorrt_workspace_gb', 1)

    def optimize(self, model: torch.nn.Module, device: torch.device,
                 example_input: Optional[torch.Tensor] = None) -> torch.nn.Module:
        """
        Apply platform-specific optimizations to model.

        Args:
            model: PyTorch model to optimize
            device: Target device (cpu, cuda, mps)
            example_input: Example input tensor for tracing/compilation

        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model for device: {device.type}")

        # Move to device first
        model = model.to(device)
        model.eval()

        # Step 1: Channels Last (universal, except CPU where it may be slower)
        if self.use_channels_last and device.type in ['cuda', 'mps']:
            logger.info("  Converting to channels_last memory format")
            model = model.to(memory_format=torch.channels_last)

        # Step 2: Mixed Precision (FP16)
        # Note: MPS FP16 support is limited and may cause slowdowns/correctness issues
        # Only enable FP16 for CUDA (NVIDIA GPUs)
        if self.precision == 'fp16' and device.type == 'cuda':
            logger.info("  Converting to FP16")
            model = model.half()
        elif self.precision == 'fp16' and device.type == 'mps':
            logger.warning("  ⚠️  FP16 disabled for MPS (limited support, use FP32)")
            self.precision = 'fp32'  # Override to FP32 for MPS

        # Step 3: Platform-specific compilation
        if self.use_compilation:
            if device.type == 'cuda':
                model = self._optimize_cuda(model, example_input)
            elif device.type == 'mps':
                model = self._optimize_mps(model, example_input)
            elif device.type == 'cpu':
                logger.info("  CPU device: skipping compilation (limited benefit)")

        logger.info("✅ Model optimization complete")
        return model

    def _optimize_cuda(self, model: torch.nn.Module,
                      example_input: Optional[torch.Tensor]) -> torch.nn.Module:
        """Apply NVIDIA CUDA-specific optimizations (TensorRT)."""
        logger.info("  Attempting TensorRT compilation (NVIDIA GPU)")

        try:
            import torch_tensorrt

            if example_input is None:
                # Create default example input for Othello (batch_size=64)
                example_input = torch.randn(64, 4, 8, 8)

            # Move to device and convert to FP16 if needed
            example_input = example_input.to(model.device)
            if self.precision == 'fp16':
                example_input = example_input.half()

            # Apply channels last if enabled
            if self.use_channels_last:
                example_input = example_input.to(memory_format=torch.channels_last)

            logger.info(f"    Compiling with TensorRT (workspace: {self.tensorrt_workspace_gb}GB)")

            # Compile with TensorRT
            compiled_model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.half} if self.precision == 'fp16' else {torch.float},
                workspace_size=self.tensorrt_workspace_gb * (1 << 30),  # Convert GB to bytes
                truncate_long_and_double=True,
                device=model.device
            )

            logger.info("    ✅ TensorRT compilation successful")
            return compiled_model

        except ImportError:
            logger.warning("    ⚠️  torch_tensorrt not installed, falling back to TorchScript")
            return self._fallback_torchscript(model, example_input)
        except Exception as e:
            logger.warning(f"    ⚠️  TensorRT compilation failed: {e}")
            logger.warning("    Falling back to TorchScript")
            return self._fallback_torchscript(model, example_input)

    def _optimize_mps(self, model: torch.nn.Module,
                     example_input: Optional[torch.Tensor]) -> torch.nn.Module:
        """Apply Apple Silicon MPS-specific optimizations (TorchScript)."""
        logger.info("  Attempting TorchScript compilation (Apple Silicon)")

        return self._fallback_torchscript(model, example_input)

    def _fallback_torchscript(self, model: torch.nn.Module,
                             example_input: Optional[torch.Tensor]) -> torch.nn.Module:
        """Fallback to TorchScript tracing."""
        try:
            # Ensure example input matches model's dtype and memory format
            if example_input is None:
                example_input = torch.randn(1, 4, 8, 8)

            # Match model's device
            example_input = example_input.to(model.device)

            # Match model's dtype (check first parameter)
            first_param = next(model.parameters())
            if first_param.dtype == torch.float16:
                example_input = example_input.half()
            elif first_param.dtype == torch.float32:
                example_input = example_input.float()

            # Match memory format
            if self.use_channels_last:
                example_input = example_input.to(memory_format=torch.channels_last)

            logger.info("    Tracing model with TorchScript")
            traced_model = torch.jit.trace(model, example_input)
            logger.info("    ✅ TorchScript tracing successful")
            return traced_model

        except Exception as e:
            logger.warning(f"    ⚠️  TorchScript tracing failed: {e}")
            logger.warning("    Using eager mode (no compilation)")
            return model


def create_inference_optimizer(config_dict: Optional[Dict[str, Any]] = None) -> InferenceOptimizer:
    """Factory function to create inference optimizer from config."""
    return InferenceOptimizer(config_dict)
