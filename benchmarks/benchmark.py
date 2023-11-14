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

import argparse

import torch

import benchmark_layer_norm
import benchmark_rms_norm
import utils


def print_scenarios() -> None:
    print("Following scenarios can be chosen:")
    print(
        ", ".join(
            [
                "layer_norm",
                "rms_norm",
            ]
        )
    )


def run_benchmark(scenario: str, show_plots: bool, dtype: torch.dtype) -> None:
    if scenario == "layer_norm":
        benchmark_layer_norm.run(show_plots, dtype)
    elif scenario == "rms_norm":
        benchmark_rms_norm.run(show_plots, dtype)
    else:
        print_scenarios()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="specify a scenario to run", type=str)
    parser.add_argument("--show-plots", action="store_true", help="show plots")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="specify a dtype to run",
        type=str,
    )
    parser.add_argument("--list", action="store_true", help="list all scenarios can be run")
    args = parser.parse_args()

    if args.list:
        print_scenarios()
    else:
        run_benchmark(
            args.scenario.replace("-", "_") if args.scenario else None,
            args.show_plots,
            utils.dtype(args.dtype),
        )


if __name__ == "__main__":
    main()
