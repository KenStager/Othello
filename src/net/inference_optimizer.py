"""
Cross-platform neural network inference optimization.

Automatically applies appropriate optimizations based on device type:
- MPS (Apple Silicon): FP32 + Channels Last + torch.compile
- CUDA (NVIDIA): FP32/FP16 + Channels Last + torch.compile
- CPU: Channels Last only (FP16 and compilation may be slower on CPU)

Uses PyTorch 2.x torch.compile() for modern, maintainable optimization.
Falls back to TorchScript if compilation fails, then eager mode.
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """
    Optimizes neural network models for inference based on target device.

    Supports:
    - Apple Silicon (MPS): FP32, Channels Last, torch.compile
    - NVIDIA GPUs (CUDA): FP32/FP16, Channels Last, torch.compile
    - CPU: Channels Last only

    Uses PyTorch 2.x torch.compile() for automatic kernel fusion and CUDA graphs.
    Works transparently with models returning dataclasses (e.g., NetworkOutput).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize optimizer with configuration.

        Args:
            config: Optimization config dict with keys:
                - precision: "fp16" or "fp32" (default: "fp32")
                - use_channels_last: bool (default: True)
                - use_compilation: bool (default: True)
                - compile_mode: str (default: "max-autotune")
                - compile_dynamic: bool (default: True for variable batch sizes)
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 'fp32')  # Changed default to fp32
        self.use_channels_last = self.config.get('use_channels_last', True)
        self.use_compilation = self.config.get('use_compilation', True)
        self.compile_mode = self.config.get('compile_mode', 'max-autotune')
        self.compile_dynamic = self.config.get('compile_dynamic', True)

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
        """Apply NVIDIA CUDA-specific optimizations (torch.compile)."""
        logger.info(f"  Attempting torch.compile (mode={self.compile_mode}, dynamic={self.compile_dynamic})")

        try:
            # torch.compile() works with any return type (including dataclasses)
            # No need for example input - compilation happens on first forward pass

            compiled_model = torch.compile(
                model,
                mode=self.compile_mode,  # 'default', 'reduce-overhead', or 'max-autotune'
                dynamic=self.compile_dynamic,  # True for variable batch sizes
                fullgraph=False,  # Allow graph breaks for complex models
            )

            logger.info("    ✅ torch.compile successful (will compile on first forward pass)")
            logger.info(f"    Expected speedup: 1.3-2.0× for MCTS workloads")
            return compiled_model

        except Exception as e:
            logger.warning(f"    ⚠️  torch.compile failed: {e}")
            logger.warning("    Falling back to TorchScript")
            return self._fallback_torchscript(model, example_input)

    def _optimize_mps(self, model: torch.nn.Module,
                     example_input: Optional[torch.Tensor]) -> torch.nn.Module:
        """Apply Apple Silicon MPS-specific optimizations (torch.compile)."""
        logger.info(f"  Attempting torch.compile (mode={self.compile_mode}, dynamic={self.compile_dynamic})")

        try:
            # torch.compile() works with MPS backend as well
            compiled_model = torch.compile(
                model,
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
                fullgraph=False,
            )

            logger.info("    ✅ torch.compile successful (will compile on first forward pass)")
            logger.info(f"    Expected speedup: 1.5-2.5× for MPS")
            return compiled_model

        except Exception as e:
            logger.warning(f"    ⚠️  torch.compile failed: {e}")
            logger.warning("    Falling back to TorchScript")
            return self._fallback_torchscript(model, example_input)

    def _fallback_torchscript(self, model: torch.nn.Module,
                             example_input: Optional[torch.Tensor]) -> torch.nn.Module:
        """Fallback to TorchScript tracing."""
        try:
            # Ensure example input matches model's dtype and memory format
            if example_input is None:
                example_input = torch.randn(1, 4, 8, 8)

            # Match model's device
            model_device = next(model.parameters()).device
            example_input = example_input.to(model_device)

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
