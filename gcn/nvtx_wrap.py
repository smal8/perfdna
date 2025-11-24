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

        # we do _orig_forward=..., _name=... because all wrapped
        # functions will end up printing the last name in the loop instead
        # due to python's closure behavior (as these functions will be run later)
        # so binding them as default arguments ensures you "freeze" the values at definition time
        def wrapped_forward(*args, _orig_forward=orig_forward, _name=name, **kwargs):
            nvtx.range_push(_name) # start a nvtx range
            try:
                return _orig_forward(*args, **kwargs)
            finally:
                nvtx.range_pop() # close the nvtx range
            
        module.forward = wrapped_forward
    
    return model

"""
We want to insert nvtx markers around each submodule's forward() call.
To do that, we replace each submodule's forward function with a wrapped version.
This is called monkey-patching.

"""