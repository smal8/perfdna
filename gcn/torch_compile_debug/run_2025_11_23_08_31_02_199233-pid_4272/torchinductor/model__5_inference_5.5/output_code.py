
# AOT ID: ['5_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/3h/c3hmogn2kx32jvrbr53mc3dyif7dk7ztnqds4zvja2uyyh3mpssu.py
# Source Nodes: [new_zeros, out, out_1, x_j], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
# new_zeros => full_default
# out => mul
# out_1 => scatter_add
# x_j => index
triton_poi_fused_index_select_mul_new_zeros_scatter_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18956
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/lh/clhzie3bmpoqlykpae277se3g5wjvfeaxouk4tseve5pevprrqoq.py
# Source Nodes: [new_zeros, out, out_1, x_j], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
# new_zeros => full_default
# out => mul
# out_1 => scatter_add
# x_j => index
triton_poi_fused_index_select_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_mul_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7)
    x0 = xindex % 7
    tmp0 = tl.load(in_ptr0 + (13264 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2708, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 2708)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 2708")
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 2708)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 2708")
    tmp12 = tl.load(in_ptr2 + (x0 + (7*tmp10)), xmask)
    tmp13 = tmp6 * tmp12
    tl.atomic_add(out_ptr0 + (x0 + (7*tmp4)), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jp/cjpfkowitxpacv2krztda6ph3mdgka7obgv23mzuijrzis4oui6a.py
# Source Nodes: [out_2], Original ATen: [aten.add]
# out_2 => add
triton_poi_fused_add_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18956
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 7
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (7, 16), (16, 1))
    assert_size_stride(arg1_1, (7, ), (1, ))
    assert_size_stride(arg2_1, (2, 13264), (13264, 1))
    assert_size_stride(arg3_1, (13264, ), (1, ))
    assert_size_stride(arg4_1, (2708, 16), (16, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2708, 7), (7, 1), torch.float32)
        # Source Nodes: [x], Original ATen: [aten.mm]
        extern_kernels.mm(arg4_1, reinterpret_tensor(arg0_1, (16, 7), (1, 16), 0), out=buf0)
        del arg0_1
        del arg4_1
        buf1 = empty_strided_cuda((2708, 7), (7, 1), torch.float32)
        # Source Nodes: [new_zeros, out, out_1, x_j], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_0.run(buf1, 18956, grid=grid(18956), stream=stream0)
        # Source Nodes: [new_zeros, out, out_1, x_j], Original ATen: [aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add]
        triton_poi_fused_index_select_mul_new_zeros_scatter_add_1.run(arg2_1, arg3_1, buf0, buf1, 92848, grid=grid(92848), stream=stream0)
        del arg2_1
        del arg3_1
        buf3 = buf0; del buf0  # reuse
        # Source Nodes: [out_2], Original ATen: [aten.add]
        triton_poi_fused_add_2.run(buf1, arg1_1, buf3, 18956, grid=grid(18956), stream=stream0)
        del arg1_1
        del buf1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((7, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 13264), (13264, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((13264, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2708, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
