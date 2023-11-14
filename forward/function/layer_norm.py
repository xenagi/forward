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

from typing import List, Optional

import functools
import torch
import triton

from forward import kernel


def layer_norm(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Applies Layer Normalization for last certain number of dimensions.

    See LayerNorm for details.
    """
    if not input.is_contiguous():
        raise ValueError("Error: A tensor must be contiguous in memory!")

    output_shape = input.shape
    x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
    input = input.view(-1, x_size)
    num_batches = input.shape[0]
    output = torch.empty_like(input)

    if weight is not None:
        weight = weight.view(x_size)
        weight_x_stride = weight.stride(0)
    else:
        weight_x_stride = 0

    if bias is not None:
        bias = bias.view(x_size)
        bias_x_stride = bias.stride(0)
    else:
        bias_x_stride = 0

    def grid(meta):
        return (num_batches,)

    kernel.layer_norm[grid](
        output,
        output.stride(0),
        output.stride(1),
        input,
        input.stride(0),
        input.stride(1),
        weight,
        weight_x_stride,
        bias,
        bias_x_stride,
        x_size,
        eps,
        triton.next_power_of_2(x_size),
    )

    return output.view(output_shape)
