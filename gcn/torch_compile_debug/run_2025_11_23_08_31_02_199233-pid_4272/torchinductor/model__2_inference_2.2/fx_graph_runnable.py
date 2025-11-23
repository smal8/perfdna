
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


torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.debug_partitioner = True



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
        full_default = torch.ops.aten.full.default([13264], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1 = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        view = torch.ops.aten.view.default(select_1, [-1])
        expand = torch.ops.aten.expand.default(view, [13264]);  view = None
        full_default_1 = torch.ops.aten.full.default([2708], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default_1, 0, expand, full_default);  full_default_1 = expand = full_default = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(scatter_add, -0.5);  scatter_add = None
        eq = torch.ops.aten.eq.Scalar(pow_1, inf)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(eq, full_default_2, pow_1);  eq = full_default_2 = pow_1 = None
        index = torch.ops.aten.index.Tensor(where, [select]);  select = None
        index_1 = torch.ops.aten.index.Tensor(where, [select_1]);  where = select_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(index, index_1);  index = index_1 = None
        return (mul_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 212224, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (2, 13264), dtype=torch.int64, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)