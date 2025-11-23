
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        permute = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm = torch.ops.aten.mm.default(arg4_1, permute);  arg4_1 = permute = None
        select = torch.ops.aten.select.int(arg2_1, 0, 1)
        select_1 = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        index = torch.ops.aten.index.Tensor(mm, [select_1]);  mm = select_1 = None
        view = torch.ops.aten.view.default(arg3_1, [-1, 1]);  arg3_1 = None
        mul = torch.ops.aten.mul.Tensor(view, index);  view = index = None
        view_1 = torch.ops.aten.view.default(select, [-1, 1]);  select = None
        expand = torch.ops.aten.expand.default(view_1, [13264, 16]);  view_1 = None
        full_default = torch.ops.aten.full.default([2708, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  full_default = expand = mul = None
        add = torch.ops.aten.add.Tensor(scatter_add, arg1_1);  scatter_add = arg1_1 = None
        return (add,)
        
def load_args(reader):
    buf0 = reader.storage(None, 91712, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 1433), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf1, (16,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 212224, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (2, 13264), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 53056, device=device(type='cuda', index=0))
    reader.tensor(buf3, (13264,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 15522256, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2708, 1433), is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)