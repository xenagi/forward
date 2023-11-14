# Copyright 2023 XenAGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.heuristics({"check_x_boundary": lambda args: args["x_size"] % args["x_block_size"]})
@triton.jit
def layer_norm(
    output_ptr: tl.tensor,
    output_batch_stride: tl.int64,
    output_x_stride: tl.int64,
    input_ptr: tl.tensor,
    input_batch_stride: tl.int64,
    input_x_stride: tl.int64,
    weight_ptr: tl.tensor,
    weight_x_stride: tl.int64,
    bias_ptr: tl.tensor,
    bias_x_stride: tl.int64,
    x_size: tl.int64,
    eps: tl.float32,
    x_block_size: tl.constexpr,
    check_x_boundary: tl.constexpr,
) -> None:
    batch = tl.program_id(0)

    input_ptr_offset = batch * input_batch_stride
    input_block_ptr = tl.make_block_ptr(
        input_ptr + input_ptr_offset,
        shape=(x_size,),
        strides=(input_x_stride,),
        offsets=(0,),
        block_shape=(x_block_size,),
        order=(0,),
    )

    if check_x_boundary:
        input = tl.load(input_block_ptr, boundary_check=(0,))
    else:
        input = tl.load(input_block_ptr)

    mean = tl.sum(input / x_size, 0)

    if check_x_boundary:
        centered_mean = tl.where(tl.arange(0, x_block_size) < x_size, input - mean, 0)
    else:
        centered_mean = input - mean

    var = tl.sum((centered_mean * centered_mean) / x_size, 0)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd

    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(x_size,),
            strides=(weight_x_stride,),
            offsets=(0,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        if check_x_boundary:
            output *= tl.load(weight_block_ptr, boundary_check=(0,))
        else:
            output *= tl.load(weight_block_ptr)

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            bias_ptr,
            shape=(x_size,),
            strides=(bias_x_stride,),
            offsets=(0,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        if check_x_boundary:
            output += tl.load(bias_block_ptr, boundary_check=(0,))
        else:
            output += tl.load(bias_block_ptr)

    output_ptr_offset = batch * output_batch_stride
    output_block_ptr = tl.make_block_ptr(
        output_ptr + output_ptr_offset,
        shape=(x_size,),
        strides=(output_x_stride,),
        offsets=(0,),
        block_shape=(x_block_size,),
        order=(0,),
    )
    output_dtype = output_block_ptr.type.element_ty

    if check_x_boundary:
        tl.store(output_block_ptr, output.to(output_dtype), boundary_check=(0,))
    else:
        tl.store(output_block_ptr, output.to(output_dtype))
