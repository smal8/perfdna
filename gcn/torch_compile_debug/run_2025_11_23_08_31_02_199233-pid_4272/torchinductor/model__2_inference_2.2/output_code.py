
# AOT ID: ['2_inference']
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


# kernel path: /tmp/torchinductor_root/hq/chqf23gbej3h273b4p3nnc63nqy7gppdaexvlyjyqww3v677gd3c.py
# Source Nodes: [deg, edge_weight, new_zeros], Original ATen: [aten.new_zeros, aten.ones, aten.scatter_add]
# deg => scatter_add
# edge_weight => full_default
# new_zeros => full_default_1
triton_poi_fused_new_zeros_ones_scatter_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_ones_scatter_add_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2708
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


# kernel path: /tmp/torchinductor_root/ib/cib5hmzdz3hvo545x7ixeujpkfrm2dzdfapk7tjdonah7pfbdsfm.py
# Source Nodes: [deg, edge_weight, new_zeros], Original ATen: [aten.new_zeros, aten.ones, aten.scatter_add]
# deg => scatter_add
# edge_weight => full_default
# new_zeros => full_default_1
triton_poi_fused_new_zeros_ones_scatter_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_ones_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13264 + x0), xmask)
    tmp1 = tl.full([XBLOCK], 2708, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 2708)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 2708")
    tmp6 = 1.0
    tl.atomic_add(out_ptr0 + (tl.broadcast_to(tmp4, [XBLOCK])), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/d5/cd55awuxh43xdathgza2ed4uqsxhogo6bl5btlq3evk4uz2copch.py
# Source Nodes: [deg_inv_sqrt, edge_weight_1, eq, getitem_3, masked_fill_, mul], Original ATen: [aten.eq, aten.index, aten.masked_fill, aten.mul, aten.pow]
# deg_inv_sqrt => pow_1
# edge_weight_1 => mul_1
# eq => eq
# getitem_3 => index_1
# masked_fill_ => full_default_2, where
# mul => index
triton_poi_fused_eq_index_masked_fill_mul_pow_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_index_masked_fill_mul_pow_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A1A999C712D012AAAD112A22ED69BAC53DFBBA6EAF7D0DB3123EC6062FEB150E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp13 = tl.load(in_ptr0 + (13264 + x0), xmask)
    tmp1 = tl.full([XBLOCK], 2708, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 2708)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 2708")
    tmp6 = tl.load(in_ptr1 + (tmp4), xmask, eviction_policy='evict_last')
    tmp7 = -0.5
    tmp8 = libdevice.pow(tmp6, tmp7)
    tmp9 = float("inf")
    tmp10 = tmp8 == tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp10, tmp11, tmp8)
    tmp14 = tmp13 + tmp1
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert(((0 <= tmp16) & (tmp16 < 2708)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 2708")
    tmp18 = tl.load(in_ptr1 + (tmp16), xmask, eviction_policy='evict_last')
    tmp19 = libdevice.pow(tmp18, tmp7)
    tmp20 = tmp19 == tmp9
    tmp21 = tl.where(tmp20, tmp11, tmp19)
    tmp22 = tmp12 * tmp21
    tl.store(out_ptr0 + (x0), tmp22, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 13264), (13264, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2708, ), (1, ), torch.float32)
        # Source Nodes: [deg, edge_weight, new_zeros], Original ATen: [aten.new_zeros, aten.ones, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_ones_scatter_add_0.run(buf0, 2708, grid=grid(2708), stream=stream0)
        # Source Nodes: [deg, edge_weight, new_zeros], Original ATen: [aten.new_zeros, aten.ones, aten.scatter_add]
        triton_poi_fused_new_zeros_ones_scatter_add_1.run(arg0_1, buf0, 13264, grid=grid(13264), stream=stream0)
        buf2 = empty_strided_cuda((13264, ), (1, ), torch.float32)
        # Source Nodes: [deg_inv_sqrt, edge_weight_1, eq, getitem_3, masked_fill_, mul], Original ATen: [aten.eq, aten.index, aten.masked_fill, aten.mul, aten.pow]
        triton_poi_fused_eq_index_masked_fill_mul_pow_2.run(arg0_1, buf0, buf2, 13264, grid=grid(13264), stream=stream0)
        del arg0_1
        del buf0
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 13264), (13264, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
