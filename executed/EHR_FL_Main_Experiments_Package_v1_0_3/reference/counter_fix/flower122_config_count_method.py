# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Verbatim method excerpt, dedented, from Flower v1.22.0 ConfigRecord.
# This reference is TEST-ONLY, not the installed class or a replacement for it.
# Source: framework/py/flwr/common/record/configrecord.py, lines 134-175.
# Upstream file Git blob SHA: c9b927c2b3d3180f6b063366f8cd5ae3bf8d6a33
from __future__ import annotations


def count_bytes(self) -> int:
    """Return number of Bytes stored in this object.

    This function counts booleans as occupying 1 Byte.
    """

    def get_var_bytes(value: ConfigScalar) -> int:
        """Return Bytes of value passed."""
        var_bytes = 0
        if isinstance(value, bool):
            var_bytes = 1
        elif isinstance(value, (int, float)):
            var_bytes = (
                8  # the profobufing represents int/floats in ConfigRecords as 64bit
            )
        if isinstance(value, (str, bytes)):
            var_bytes = len(value)
        if var_bytes == 0:
            raise ValueError(
                "Config values must be either `bool`, `int`, `float`, "
                "`str`, or `bytes`"
            )
        return var_bytes

    num_bytes = 0

    for k, v in self.items():
        if isinstance(v, list):
            if isinstance(v[0], (bytes, str)):
                # not all str are of equal length necessarily
                # for both the footprint of each element is 1 Byte
                num_bytes += int(sum(len(s) for s in v))  # type: ignore
            else:
                num_bytes += get_var_bytes(v[0]) * len(v)
        else:
            num_bytes += get_var_bytes(v)

        # We also count the bytes footprint of the keys
        num_bytes += len(k)

    return num_bytes
