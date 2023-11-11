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

from typing import Optional, Sequence

import pytest
import torch

import forward


def rms_norm(
    input: torch.Tensor, p: float, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-08
) -> torch.Tensor:
    x_size = input.shape[-1]

    if p < 0.0 or p > 1.0:
        norm = input.norm(2, dim=-1, keepdim=True)
        partial_size = x_size
    else:
        partial_size = int(x_size * p)
        partial_input, _ = torch.split(input, [partial_size, x_size - partial_size], dim=-1)
        norm = partial_input.norm(2, dim=-1, keepdim=True)

    rms = norm * partial_size ** (-1.0 / 2)
    output = input / (rms + eps)
    output *= weight

    if bias is not None:
        output += bias

    return output


@pytest.mark.parametrize("input_shape, p", [([1000, 10], 0.5), ([10, 1000], 1.0)])
def test_function(input_shape: Sequence[int], p: float, device) -> None:
    input = torch.randn(input_shape, device=device)
    weight = torch.randn(input_shape[-1], device=device)
    bias = torch.randn(input_shape[-1], device=device)

    a = rms_norm(input, p, weight)
    b = forward.function.rms_norm(input, p, weight)

    assert torch.allclose(a, b, atol=1e-05)

    a = rms_norm(input, p, weight, bias)
    b = forward.function.rms_norm(input, p, weight, bias)

    assert torch.allclose(a, b, atol=1e-05)


@pytest.mark.parametrize("input_shape, p", [([2, 16], 0.3)])
def test_rms_norm(input_shape: Sequence[int], p: float, device, dtype) -> None:
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(input_shape, **factory_kwargs)

    module = forward.RMSNorm(input_shape[-1], p, **factory_kwargs)
    output = module(input)

    assert output is not None
    assert output.dtype is dtype

    module = forward.RMSNorm(input_shape[-1], p, bias=True, **factory_kwargs)
    output = module(input)

    assert output is not None
    assert output.dtype is dtype
