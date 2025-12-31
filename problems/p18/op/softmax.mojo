from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = Layout.row_major(SIZE)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 31 lines)
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    shared = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    xi = LayoutTensor[dtype, layout, ImmutAnyOrigin].element_type(0)
    if global_i < UInt(input_size):
        xi = input[global_i]
        shared[local_i] = xi
    barrier()
    i: LayoutTensor[dtype, layout, ImmutAnyOrigin].element_type = 1
    while i < input_size:
        if global_i < UInt(input_size) and (local_i - i) >= 0:
            if shared[local_i - i] > shared[local_i]:
                shared[local_i] = shared[local_i - i]
        i = 2 * i
        barrier()
    numerator: LayoutTensor[dtype, layout, MutAnyOrigin].element_type = 0
    if global_i < UInt(input_size):
        maxed = shared[UInt(input_size) - 1]
        numerator = exp(xi - maxed)
    barrier()

    if global_i < UInt(input_size):
        shared[local_i] += numerator
    barrier()
    i = 1
    while i < UInt(input_size):
        if global_i < UInt(input_size) and (local_i - i) >= 0:
            shared[local_i] += shared[local_i - i]
        i = 2 * i
        barrier()
    denominator = shared[UInt(input_size) - 1]

    if global_i < UInt(input_size):
        output[global_i] = numerator / denominator


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel_solution
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    var max_val: Scalar[dtype] = min_finite[dtype]()
    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    var sum_exp: Scalar[dtype] = 0.0
    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel_solution

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=1,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
