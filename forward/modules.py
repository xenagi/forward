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

from typing import Tuple, Union

import torch

from forward import function, utils


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        p: float = -1.0,
        eps: float = 1e-05,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """
        Applies Root Mean Square Layer Normalization to an input.

        Args:
            normalized_shape: input shape from an expected input of size
            p: partial Root Mean Square Layer Normalization, valid value [0, 1] otherwise it's disabled
            eps: a value added to the denominator for numerical stability
            bias: a boolean value that when set to True, this module has learnable bias parameters.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.normalized_shape = normalized_shape
        self.p = p
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Applies Root Mean Square Layer Normalization to an input.

        Args:
            input: an input

        Returns:
            an output with the same shape as an input
        """
        return function.rms_norm(input, self.p, self.weight, self.bias, self.eps)

    def reset_parameters(self) -> None:
        """
        Reset parameters of the module.
        """
        utils.fill(self.weight, 1.0)

        if self.bias is not None:
            utils.zero(self.bias)

    def extra_repr(self) -> str:
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{tuple(self.normalized_shape)}, "
            f"p={self.p}, "
            f"eps={self.eps}, "
            f"bias={self.bias}, "
            f"backend=forward"
        )
