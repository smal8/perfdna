import torch
import torch.cuda.nvtx as nvtx

def add_nvtx(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wraps every submodule's forward() in an NVTX range so that
    the layers show up as regions in the CUDA trace.

    Usage:
        model = GCN()
        add_nvtx(model)
        # run as normal
    """

    # named_modules gives ("", model), ("conv1", module), ("relu", module), ("conv2", module)
    for name, module in model.named_modules():
        if name == "":
            continue

        orig_forward = module.forward # keep original

        def wrapped_forward(*args, _orig_forward=orig_forward, _name=name, **kwargs):
            nvtx.range_push(_name)
            try:
                return _orig_forward(*args, **kwargs)
            finally:
                nvtx.range_pop()
            
        module.forward = wrapped_forward
    
    return model