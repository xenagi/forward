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

from typing import Any, Callable, Dict, List

import torch
import triton


def dtype(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    elif value == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unable to convert the given input: '{value}'.")


def make_benchmark(title: str, x_names: List[str], x_vals: Any, args: Dict[str, Any]) -> triton.testing.Benchmark:
    return triton.testing.Benchmark(
        x_names,
        x_vals,
        "backend",
        ["torch", "forward"],
        ["torch", "forward"],
        title,
        args,
        ylabel="milliseconds",
    )


def report(title: str, x_names: List[str], x_vals: Any, args: Dict[str, Any] = None) -> Callable:
    return triton.testing.perf_report([make_benchmark(title, x_names, x_vals, args if args is not None else {})])
