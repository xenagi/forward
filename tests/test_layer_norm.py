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

from typing import Sequence

import pytest
import torch

import forward


@pytest.mark.parametrize("input_shape", [([1000, 10]), ([10, 1000])])
def test_function(input_shape: Sequence[int], device) -> None:
    input = torch.randn(input_shape, device=device)
    normalized_shape = [input_shape[-1]]
    weight = torch.ones(normalized_shape, device=device)
    bias = torch.randn(normalized_shape, device=device)

    a = torch.nn.functional.layer_norm(input, normalized_shape)
    b = forward.function.layer_norm(input, normalized_shape)

    assert torch.allclose(a, b, atol=1e-05)

    a = torch.nn.functional.layer_norm(input, normalized_shape, weight)
    b = forward.function.layer_norm(input, normalized_shape, weight)

    assert torch.allclose(a, b, atol=1e-05)

    a = torch.nn.functional.layer_norm(input, normalized_shape, bias=bias)
    b = forward.function.layer_norm(input, normalized_shape, bias=bias)

    assert torch.allclose(a, b, atol=1e-05)

    a = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias)
    b = forward.function.layer_norm(input, normalized_shape, weight, bias)

    assert torch.allclose(a, b, atol=1e-05)


@pytest.mark.parametrize("input_shape", [([10, 1000])])
def test_rms_norm(input_shape: Sequence[int], device, dtype) -> None:
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(input_shape, **factory_kwargs)
    normalized_shape = [input_shape[-1]]

    module = forward.LayerNorm(normalized_shape, **factory_kwargs)
    output = module(input)

    assert output is not None
    assert output.dtype is dtype

    module = forward.LayerNorm(normalized_shape, elementwise_affine=True, **factory_kwargs)
    output = module(input)

    assert output is not None
    assert output.dtype is dtype
