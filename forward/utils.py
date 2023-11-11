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

import torch


def fill(input: torch.Tensor, value: float) -> torch.Tensor:
    with torch.no_grad():
        return input.fill_(value)


def zero(input: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return input.zero_()
