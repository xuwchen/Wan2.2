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


def to_local_if_dtensor(tensor):
    """
    Convert a DTensor to a local tensor (following Megatron-LM's approach).
    
    Args:
        tensor: A tensor that may be a DTensor.
    Returns:
        torch.Tensor: The local tensor.
    """
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(tensor, DTensor):
            return tensor._local_tensor
    except ImportError:
        pass
    return tensor


class TEMainGradBuffer:
    """
    Gradient buffer manager for TE Linear layers with Megatron-FSDP.
    
    This class follows Megatron-LM's approach:
    1. If Megatron-FSDP already set up main_grad for weights, use those directly
    2. Otherwise, create a contiguous buffer (fallback for FSDP without Megatron main_grad)
    
    Key insight from Megatron-LM:
    - When weight is a DTensor (FSDP sharded), use to_local_if_dtensor() to get local shape
    - The main_grad buffer should be sized for LOCAL shards, not global tensors
    - sync_to_weight_grad() should copy to weight.grad._local_tensor for DTensor
    
    Usage:
        # After FSDP wrapping
        grad_buffer = TEMainGradBuffer(model)
        
        # Before first microbatch of each step
        grad_buffer.zero_grad()
        
        # Training loop works with fuse_wgrad_accumulation=True
        # No need to call sync_to_weight_grad() when using Megatron-FSDP
    """
    
    def __init__(self, model: nn.Module, alignment: int = 128):
        """
        Initialize the gradient buffer manager.
        
        Args:
            model: Model containing TE Linear layers (after FSDP wrapping)
            alignment: Memory alignment in elements (default: 128 for cuBLAS)
        """
        self.model = model
        self.alignment = alignment
        self.buffer = None  # Only used for fallback (non-FSDP)
        self.weight_info = []  # List of (name, weight, uses_fsdp_main_grad)
        self.uses_fsdp_main_grad = False
        
        if not TE_AVAILABLE:
            print("Warning: TransformerEngine not available")
            return
        
        self._setup_buffer()
    
    def _align_offset(self, offset: int) -> int:
        """Align offset to the required boundary."""
        return ((offset + self.alignment - 1) // self.alignment) * self.alignment
    
    def _setup_buffer(self):
        """
        Set up TE Linear weights to use appropriate main_grad.
        
        Megatron-FSDP sets up get_main_grad() method on each parameter.
        We check for this and use it directly if available.
        Otherwise, we create our own buffer (fallback for non-FSDP cases).
        """
        weights_with_fsdp_main_grad = []
        weights_without_main_grad = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, te.Linear):
                if getattr(module, 'fuse_wgrad_accumulation', False):
                    weight = module.weight
                    
                    # Check if Megatron-FSDP already set up get_main_grad
                    # Megatron-FSDP sets get_main_grad method AND _gbuf/_item_id attributes
                    has_fsdp_main_grad = (
                        hasattr(weight, 'get_main_grad') and 
                        hasattr(weight, '_gbuf') and 
                        hasattr(weight, '_item_id')
                    )
                    
                    if has_fsdp_main_grad:
                        weights_with_fsdp_main_grad.append((name, weight))
                    else:
                        weights_without_main_grad.append((name, weight))
        
        # Prefer using Megatron-FSDP's main_grad if available
        if weights_with_fsdp_main_grad:
            self.uses_fsdp_main_grad = True
            print(f"Found {len(weights_with_fsdp_main_grad)} TE Linear layers with Megatron-FSDP get_main_grad")
            
            for name, weight in weights_with_fsdp_main_grad:
                # Megatron-FSDP already set __fsdp_param__ = True and get_main_grad
                # Just record the weight, don't override anything
                self.weight_info.append((name, weight, True))
            
            print(f"Using Megatron-FSDP's get_main_grad for {len(weights_with_fsdp_main_grad)} TE Linear layers")
            print("  (No separate buffer needed - gradients accumulate in FSDP's grad buffer)")
            
            return
        
        # Fallback: create our own contiguous buffer (for non-FSDP or testing)
        if not weights_without_main_grad:
            print("No TE Linear layers with fuse_wgrad_accumulation found")
            return
        
        print("Megatron-FSDP main_grad not found, creating standalone buffer (fallback mode)")
        
        # Calculate total size needed
        # IMPORTANT: Use LOCAL shard size for DTensor (FSDP sharded weights)
        # This follows Megatron-LM's to_local_if_dtensor() approach
        total_size = 0
        weights_to_setup = []
        
        for name, weight in weights_without_main_grad:
            # Get local tensor for DTensor (FSDP sharded)
            local_weight = to_local_if_dtensor(weight)
            local_numel = local_weight.numel()
            local_shape = local_weight.shape
            aligned_offset = self._align_offset(total_size)
            
            weights_to_setup.append({
                'name': name,
                'weight': weight,
                'offset': aligned_offset,
                'numel': local_numel,  # Use LOCAL size
                'shape': local_shape,  # Use LOCAL shape
            })
            
            total_size = aligned_offset + local_numel
        
        # Get dtype and device from first weight (use local tensor)
        first_local = to_local_if_dtensor(weights_to_setup[0]['weight'])
        dtype = first_local.dtype
        device = first_local.device
        
        # Align total size
        total_size = self._align_offset(total_size)
        
        # Allocate contiguous buffer
        self.buffer = torch.zeros(total_size, dtype=dtype, device=device)
        
        buffer_mb = total_size * self.buffer.element_size() / 1024 / 1024
        print(f"Allocated fallback main_grad buffer: {total_size} elements ({buffer_mb:.2f} MB)")
        
        # Set up get_main_grad for each weight
        for info in weights_to_setup:
            weight = info['weight']
            offset = info['offset']
            numel = info['numel']
            shape = info['shape']
            name = info['name']
            
            # Store info
            self.weight_info.append((name, weight, False))
            
            # Set __fsdp_param__ = True so TE uses get_main_grad() path
            weight.__fsdp_param__ = True
            
            # Create get_main_grad method that returns a VIEW into the contiguous buffer
            def make_getter(buf, off, num, shp):
                def getter():
                    view = buf[off:off + num].view(shp)
                    return view
                return getter
            
            weight.get_main_grad = make_getter(self.buffer, offset, numel, shape)
        
        print(f"Configured {len(weights_to_setup)} TE Linear layers with fallback main_grad buffer")
    
    def zero_grad(self):
        """
        Zero the gradient buffer. Call at the start of each step.
        
        When using Megatron-FSDP main_grad, this zeros those buffers via get_main_grad().
        When using fallback mode, this zeros our standalone buffer.
        """
        if self.uses_fsdp_main_grad:
            # Zero each weight's main_grad (managed by Megatron-FSDP)
            # Use get_main_grad() which returns a view into FSDP's grad buffer
            for name, weight, _ in self.weight_info:
                if hasattr(weight, 'get_main_grad'):
                    try:
                        main_grad = weight.get_main_grad()
                        main_grad.zero_()
                    except Exception:
                        # If get_main_grad fails (e.g., buffer not yet allocated),
                        # it's okay - the buffer will be zeroed when allocated
                        pass
        elif self.buffer is not None:
            # Zero our standalone buffer
            self.buffer.zero_()
    
    def sync_to_weight_grad(self):
        """
        Sync main_grad to weight.grad for optimizer.
        
        When using Megatron-FSDP, this is NOT needed because:
        - Megatron-FSDP's optimizer reads from main_grad directly
        - The gradient reduce happens on main_grad
        
        This method is for fallback mode (FSDP without Megatron main_grad).
        Following Megatron-LM's approach: use to_local_if_dtensor for DTensor.
        """
        if self.uses_fsdp_main_grad:
            # When using Megatron-FSDP, the optimizer uses main_grad directly
            # No sync needed - just return
            return
        
        if self.buffer is None:
            return
        
        # Fallback mode: copy from our buffer to weight.grad
        # For DTensor (FSDP sharded), we need to copy to _local_tensor
        for name, weight, _ in self.weight_info:
            main_grad = weight.get_main_grad()
            
            # Check if weight is DTensor
            try:
                from torch.distributed.tensor import DTensor
                is_dtensor = isinstance(weight, DTensor)
            except ImportError:
                is_dtensor = False
            
            if weight.grad is None:
                if is_dtensor:
                    # For DTensor, create a zeros_like to get proper DTensor structure
                    # then copy main_grad into its local tensor
                    weight.grad = torch.zeros_like(weight)
                    weight.grad._local_tensor.copy_(main_grad)
                else:
                    # For regular tensor, just clone
                    weight.grad = main_grad.clone()
            else:
                # Get local grad tensor for DTensor (following Megatron-LM)
                local_grad = to_local_if_dtensor(weight.grad)
                local_grad.copy_(main_grad)
    
    def get_buffer(self) -> torch.Tensor:
        """Get the underlying contiguous buffer tensor (fallback mode only)."""
        return self.buffer
    
    def get_total_grad_norm(self, norm_type: float = 2.0) -> torch.Tensor:
        """Compute the gradient norm across all TE Linear weights."""
        if self.uses_fsdp_main_grad:
            # Compute norm across all main_grads
            total_norm = torch.tensor(0.0, device='cuda')
            for name, weight, _ in self.weight_info:
                if hasattr(weight, 'main_grad') and weight.main_grad is not None:
                    total_norm += torch.norm(weight.main_grad, p=norm_type) ** norm_type
            return total_norm ** (1.0 / norm_type)
        elif self.buffer is not None:
            return torch.norm(self.buffer, p=norm_type)
        return torch.tensor(0.0)
    
    def num_weights(self) -> int:
        """Return the number of weights managed by this buffer."""
        return len(self.weight_info)


def setup_main_grad_for_te_linear(model: nn.Module) -> TEMainGradBuffer:
    """
    Set up TE Linear layers for gradient accumulation with a contiguous buffer.
    
    This creates a contiguous gradient buffer similar to Megatron-FSDP's
    param_and_grad_buffer, which satisfies cuBLAS GEMM alignment requirements.
    
    This should be called AFTER FSDP wrapping.
    
    Args:
        model: The model (after FSDP wrapping)
    
    Returns:
        TEMainGradBuffer object managing the gradient buffer
    """
    buffer = TEMainGradBuffer(model)
    
    # Monkey-patch TE Linear to use torch.matmul instead of cuBLAS for wgrad
    # This is necessary because cuBLAS GEMM accumulation mode (beta=1) is
    # incompatible with FSDP tensor layouts
    _patch_te_linear_for_fsdp()
    
    return buffer


_TE_PATCHED = False
_ORIGINAL_GENERAL_GEMM = None

def _patch_te_linear_for_fsdp():
    """
    Monkey-patch TE Linear's backward to use torch.matmul for wgrad when
    __fsdp_param__ is set. This avoids the cuBLAS_STATUS_NOT_SUPPORTED error.
    
    We need to patch multiple locations:
    1. transformer_engine.pytorch.cpp_extensions.gemm.general_gemm
    2. transformer_engine.pytorch.module.linear.general_gemm (the imported reference)
    """
    global _TE_PATCHED, _ORIGINAL_GENERAL_GEMM
    if _TE_PATCHED or not TE_AVAILABLE:
        return
    
    try:
        from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
        from transformer_engine.pytorch.module import linear as te_linear_module
        
        # Save original general_gemm
        _ORIGINAL_GENERAL_GEMM = te_gemm.general_gemm
        
        def patched_general_gemm(A, B, out_dtype=None, quantization_params=None,
                                  gelu=False, gelu_in=None, alpha=1.0, beta=None,
                                  accumulate=False, layout="TN", out=None, bias=None,
                                  use_split_accumulator=False, grad=False, ub=None,
                                  ub_type=None, extra_output=None, bulk_overlap=False):
            """
            Patched general_gemm that uses torch.matmul when:
            1. out is provided (main_grad buffer for wgrad)
            2. grad=True (gradient computation)
            
            This avoids cuBLAS_STATUS_NOT_SUPPORTED errors with FSDP.
            cuBLAS often fails with non-standard tensor layouts even without accumulate.
            """
            # When out is provided and this is gradient computation, use torch.matmul
            # This handles both first microbatch (accumulate=False) and subsequent ones (accumulate=True)
            use_torch_matmul = (
                out is not None and
                grad  # Only for gradient computation (wgrad)
            )
            
            if use_torch_matmul:
                # Use torch.matmul instead of cuBLAS
                # For wgrad: dW = dY^T @ X (layout="NT")
                # A = input (X), B = grad_output (dY)
                
                # Flatten 3D tensors to 2D for matmul
                # Input may be [batch, seq, hidden] -> [batch*seq, hidden]
                A_2d = A.reshape(-1, A.shape[-1]) if A.dim() == 3 else A
                B_2d = B.reshape(-1, B.shape[-1]) if B.dim() == 3 else B
                
                if layout == "NT":
                    # dW = B^T @ A -> [out_features, batch*seq] @ [batch*seq, in_features]
                    result = torch.matmul(B_2d.t(), A_2d)
                elif layout == "TN":
                    # dW = A^T @ B -> [in_features, batch*seq] @ [batch*seq, out_features]
                    result = torch.matmul(A_2d.t(), B_2d)
                else:
                    # NN layout
                    result = torch.matmul(A_2d, B_2d)
                
                # Handle accumulate mode
                # With overwrite_main_grad=False, TE passes accumulate=True (when is_first_microbatch=None)
                # The buffer is zeroed at start of each step via te_grad_buffer.zero_grad()
                # So we always use add_ to accumulate gradients across microbatches.
                if accumulate:
                    out.add_(result)
                else:
                    out.copy_(result)
                
                # Handle bias gradient if needed
                grad_bias = None
                if bias is not None:
                    grad_bias = B.sum(dim=0) if B.dim() == 2 else B.sum(dim=tuple(range(B.dim()-1)))
                
                return out, grad_bias, None, None
            
            # Fall back to original cuBLAS implementation
            return _ORIGINAL_GENERAL_GEMM(
                A, B, out_dtype=out_dtype, quantization_params=quantization_params,
                gelu=gelu, gelu_in=gelu_in, alpha=alpha, beta=beta,
                accumulate=accumulate, layout=layout, out=out, bias=bias,
                use_split_accumulator=use_split_accumulator, grad=grad, ub=ub,
                ub_type=ub_type, extra_output=extra_output, bulk_overlap=bulk_overlap
            )
        
        # Apply patch to BOTH locations
        # 1. The source module
        te_gemm.general_gemm = patched_general_gemm
        # 2. The imported reference in linear.py module
        te_linear_module.general_gemm = patched_general_gemm
        
        _TE_PATCHED = True
        print("Patched TE general_gemm in both gemm.py and linear.py for FSDP-compatible wgrad accumulation")
        
    except Exception as e:
        import traceback
        print(f"Warning: Failed to patch TE Linear for FSDP: {e}")
        traceback.print_exc()


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



