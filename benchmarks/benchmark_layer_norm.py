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

from typing import Any

import torch
import triton

import forward
import utils


@utils.report("layer_norm", ["x_size"], [512 * i for i in range(1, 11)])
def bench_rms_norm(x_size: int, dtype: torch.dtype, backend: str) -> Any:
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(4096, x_size, **factory_kwargs)
    normalized_shape = [x_size]

    if backend == "torch":
        return triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(input, normalized_shape))
    else:
        return triton.testing.do_bench(lambda: forward.function.layer_norm(input, normalized_shape))


def run(show_plots: bool, dtype: torch.dtype) -> None:
    bench_rms_norm.run(print_data=True, show_plots=show_plots, dtype=dtype)
