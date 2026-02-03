"""
Utility functions to replace nn.Linear with TransformerEngine Linear
for gradient_accumulation_fusion support.
"""

import torch
import torch.nn as nn

# TransformerEngine should be installed system-wide
try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TransformerEngine not available: {e}")
    TE_AVAILABLE = False


def test_te_linear_gemm(in_features: int, out_features: int, batch_seq_len: int, 
                        fuse_wgrad: bool = True, device="cuda", dtype=torch.bfloat16):
    """
    Test if TE Linear can run forward+backward with given dimensions.
    Returns True if successful, False if cuBLAS error occurs.
    """
    if not TE_AVAILABLE:
        return False
    
    try:
        linear = te.Linear(in_features, out_features, bias=True, 
                          fuse_wgrad_accumulation=fuse_wgrad).to(device, dtype)
        x = torch.randn(batch_seq_len, in_features, device=device, dtype=dtype, requires_grad=True)
        y = linear(x)
        loss = y.sum()
        loss.backward()
        return True
    except RuntimeError as e:
        if "cuBLAS GEMM" in str(e):
            return False
        raise


def replace_linear_with_te(
    model: nn.Module,
    fuse_wgrad_accumulation: bool = True,
    skip_modules: list = None,
    alignment: int = 16,  # cuBLAS GEMM alignment requirement
) -> nn.Module:
    """
    Replace nn.Linear modules with TransformerEngine Linear.
    Only replaces layers where both in_features and out_features meet alignment requirements.
    
    Args:
        model: The model to modify
        fuse_wgrad_accumulation: Enable gradient accumulation fusion
        skip_modules: List of module name patterns to skip (e.g., ['t5', 'vae'])
        alignment: Dimension alignment requirement (default: 16 for bf16/fp16)
    
    Returns:
        Modified model with TE Linear layers
    """
    if not TE_AVAILABLE:
        print("Warning: TransformerEngine not available, skipping replacement")
        return model
    
    skip_modules = skip_modules or []
    replaced_count = 0
    skipped_alignment = 0
    skipped_pattern = 0
    replaced_dims = []  # Track dimensions of replaced layers
    
    def should_skip(name: str) -> bool:
        return any(skip in name.lower() for skip in skip_modules)
    
    def is_aligned(in_features: int, out_features: int) -> bool:
        """Check if dimensions meet cuBLAS GEMM alignment requirements."""
        return (in_features % alignment == 0) and (out_features % alignment == 0)
    
    def replace_module(parent: nn.Module, name: str, module: nn.Module, full_name: str):
        nonlocal replaced_count, skipped_alignment
        
        if isinstance(module, nn.Linear):
            # Check alignment requirements
            if not is_aligned(module.in_features, module.out_features):
                print(f"  SKIP (alignment): {full_name} in={module.in_features}, out={module.out_features}")
                skipped_alignment += 1
                return False
            
            # Create TE Linear with same configuration
            te_linear = te.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )
            
            # Copy weights
            with torch.no_grad():
                te_linear.weight.copy_(module.weight)
                if module.bias is not None:
                    te_linear.bias.copy_(module.bias)
            
            # Move to same device/dtype
            te_linear = te_linear.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            
            # Create main_grad attribute for fuse_wgrad_accumulation
            # This is required by TE Linear when fuse_wgrad_accumulation=True
            if fuse_wgrad_accumulation:
                te_linear.weight.main_grad = torch.zeros_like(te_linear.weight)
            
            setattr(parent, name, te_linear)
            replaced_count += 1
            replaced_dims.append((full_name, module.in_features, module.out_features))
            return True
        return False
    
    def recursive_replace(module: nn.Module, prefix: str = ""):
        nonlocal skipped_pattern
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if should_skip(full_name):
                if isinstance(child, nn.Linear):
                    skipped_pattern += 1
                continue
            
            if isinstance(child, nn.Linear):
                replace_module(module, name, child, full_name)
            else:
                recursive_replace(child, full_name)
    
    recursive_replace(model)
    
    # Print summary and unique dimensions
    print(f"TE Linear replacement: {replaced_count} replaced, "
          f"{skipped_alignment} skipped (alignment), {skipped_pattern} skipped (pattern)")
    
    # Print unique dimension combinations
    unique_dims = set((in_f, out_f) for _, in_f, out_f in replaced_dims)
    print(f"Unique Linear dimensions replaced (in, out): {sorted(unique_dims)}")
    
    return model


def get_te_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    fuse_wgrad_accumulation: bool = True,
    device=None,
    dtype=None,
) -> nn.Module:
    """
    Get a Linear module - TE Linear if available, otherwise nn.Linear.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        fuse_wgrad_accumulation: Enable gradient accumulation fusion (TE only)
        device: Device to place the module
        dtype: Data type for parameters
    
    Returns:
        Linear module
    """
    if TE_AVAILABLE:
        linear = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )
    else:
        linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
    
    if device is not None or dtype is not None:
        linear = linear.to(device=device, dtype=dtype)
    
    return linear


# Expose TE availability check
def is_te_available() -> bool:
    return TE_AVAILABLE


def setup_main_grad_for_te_linear(model: nn.Module) -> int:
    """
    Set up TE Linear layers for gradient accumulation with Megatron-FSDP.
    
    This should be called AFTER FSDP wrapping. Since Megatron-FSDP doesn't 
    automatically set up get_main_grad for TE Linear layers, we manually
    create the main_grad buffer and set __fsdp_param__ = True to make TE
    use torch.matmul instead of cuBLAS accumulation mode.
    
    Args:
        model: The model (after FSDP wrapping)
    
    Returns:
        Number of TE Linear layers configured
    """
    if not TE_AVAILABLE:
        return 0
    
    count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, te.Linear):
            if getattr(module, 'fuse_wgrad_accumulation', False):
                weight = module.weight
                
                # Create a contiguous main_grad buffer
                if not hasattr(weight, 'main_grad') or weight.main_grad is None:
                    weight.main_grad = torch.zeros(
                        weight.shape,
                        dtype=weight.dtype,
                        device=weight.device,
                    ).contiguous()
                
                # Set __fsdp_param__ = True to trigger TE's torch.matmul path
                # This avoids the cuBLAS accumulation mode (beta=1) which is
                # incompatible with FSDP tensor layouts
                weight.__fsdp_param__ = True
                
                # Create get_main_grad method that returns the main_grad buffer
                # TE will call this in backward to get the output buffer for wgrad
                def make_getter(w):
                    return lambda: w.main_grad
                weight.get_main_grad = make_getter(weight)
                
                count += 1
    
    print(f"Configured {count} TE Linear layers with manual FSDP-compatible main_grad")
    return count


class TEGradientAccumulator:
    """
    Manual gradient accumulation for TE Linear with FSDP.
    
    Since cuBLAS beta=1 mode is incompatible with FSDP tensors, we use:
    1. overwrite_main_grad=True (beta=0 mode, FSDP compatible)
    2. Manual accumulation: after each backward, add main_grad to accumulated_grad
    
    Usage:
        accumulator = TEGradientAccumulator(model)
        for microbatch_idx in range(num_microbatches):
            accumulator.pre_backward(is_first=(microbatch_idx == 0))
            loss.backward()
            accumulator.post_backward()
        accumulator.finalize()  # Copy accumulated grads to weight.grad
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.accumulated_grads = {}
        self._collect_te_linears()
    
    def _collect_te_linears(self):
        """Find all TE Linear layers with fuse_wgrad_accumulation."""
        self.te_linear_weights = {}
        if not TE_AVAILABLE:
            return
        
        for name, module in self.model.named_modules():
            if isinstance(module, te.Linear):
                if getattr(module, 'fuse_wgrad_accumulation', False):
                    self.te_linear_weights[name] = module.weight
    
    def pre_backward(self, is_first: bool = False):
        """Call before each backward pass."""
        if is_first:
            # Zero accumulated grads on first microbatch
            self.accumulated_grads.clear()
            for name, weight in self.te_linear_weights.items():
                if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                    weight.main_grad.zero_()
    
    def post_backward(self):
        """Call after each backward pass to accumulate gradients."""
        for name, weight in self.te_linear_weights.items():
            if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                if name not in self.accumulated_grads:
                    # First microbatch - just copy
                    self.accumulated_grads[name] = weight.main_grad.clone()
                else:
                    # Subsequent microbatches - accumulate
                    self.accumulated_grads[name].add_(weight.main_grad)
    
    def finalize(self):
        """Copy accumulated gradients to weight.grad for optimizer."""
        for name, weight in self.te_linear_weights.items():
            if name in self.accumulated_grads:
                if weight.grad is None:
                    weight.grad = self.accumulated_grads[name].clone()
                else:
                    weight.grad.copy_(self.accumulated_grads[name])
    
    def clear(self):
        """Clear accumulated gradients."""
        self.accumulated_grads.clear()
        for name, weight in self.te_linear_weights.items():
            if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                weight.main_grad.zero_()



