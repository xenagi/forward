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

from typing import Any, Optional

import torch
import triton

import forward
import utils


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


@utils.report("rms_norm", ["x_size"], [512 * i for i in range(1, 11)])
def bench_rms_norm(x_size: int, dtype: torch.dtype, backend: str) -> Any:
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(8, 512, x_size, **factory_kwargs)
    weight = torch.randn(x_size, **factory_kwargs)

    if backend == "torch":
        return triton.testing.do_bench(lambda: rms_norm(input, 1.0, weight))
    else:
        return triton.testing.do_bench(lambda: forward.function.rms_norm(input, 1.0, weight))


def run(show_plots: bool, dtype: torch.dtype) -> None:
    bench_rms_norm.run(print_data=True, show_plots=show_plots, dtype=dtype)
