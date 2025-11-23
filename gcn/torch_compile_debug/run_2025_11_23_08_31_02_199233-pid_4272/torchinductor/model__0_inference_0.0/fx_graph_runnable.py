
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.0+cu121
# torch cuda version: 12.1
# torch git version: e4ee3be4063b7c430974252fdf7db42273388d86


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Jun__6_02:18:23_PDT_2024 
# Cuda compilation tools, release 12.5, V12.5.82 
# Build cuda_12.5.r12.5/compiler.34385749_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        select = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1 = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        ne = torch.ops.aten.ne.Tensor(select, select_1);  select = select_1 = None
        iota = torch.ops.prims.iota.default(2708, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view = torch.ops.aten.view.default(iota, [1, -1]);  iota = None
        repeat = torch.ops.aten.repeat.default(view, [2, 1]);  view = None
        return (ne, repeat)
        
def load_args(reader):
    buf0 = reader.storage(None, 168896, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 10556), dtype=torch.int64, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)