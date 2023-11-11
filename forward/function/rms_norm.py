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

from typing import Optional

import torch
import triton

from forward import kernel


def rms_norm(
    input: torch.Tensor, p: float, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-08
) -> torch.Tensor:
    """
    Applies Root Mean Square Layer Normalization to an input.

    See RMSNorm for details.
    """
    if not input.is_contiguous():
        raise ValueError("Error: A tensor must be contiguous in memory!")

    output_shape = input.shape
    input = input.view(-1, input.shape[-1])
    num_batches, x_size = input.shape
    output = torch.empty_like(input)

    def grid(meta):
        return (num_batches,)

    kernel.rms_norm[grid](
        output,
        output.stride(0),
        output.stride(1),
        input,
        input.stride(0),
        input.stride(1),
        weight,
        weight.stride(0),
        bias,
        bias.stride(0) if bias is not None else 0,
        x_size,
        x_size if p < 0.0 or p > 1.0 else x_size * p,
        eps,
        triton.next_power_of_2(x_size),
    )

    return output.view(output_shape)
