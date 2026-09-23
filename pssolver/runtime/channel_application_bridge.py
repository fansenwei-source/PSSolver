"""Narrow P7.5 bridge from the Channel application to its opt-in runtime."""

from __future__ import annotations

from pssolver.experimental.channel_compiled_v2 import (
    build_channel_compiled_v2_runtime,
)
from pssolver.runtime.channel_active_nematics import ChannelRuntimeBuildRequest


def build_package_compiled_channel_runtime(
    request: ChannelRuntimeBuildRequest,
) -> object:
    if not isinstance(request, ChannelRuntimeBuildRequest):
        raise TypeError("request must be a ChannelRuntimeBuildRequest")
    return build_channel_compiled_v2_runtime(
        request.run_spec,
        initial_q=request.initial_q,
        device=request.device,
    )


__all__ = ["build_package_compiled_channel_runtime"]
