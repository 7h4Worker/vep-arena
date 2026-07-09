"""Decision-channel information metrics for VEP/SSVEP experiments."""

from vep_arena.channel.capacity import (
    BAResult,
    ClosedCapacityResult,
    capacity_ba,
    capacity_binary_closed,
    capacity_c0,
    capacity_c1,
    capacity_c2_closed,
    conditional_entropy_rows,
    mutual_info,
    mutual_info_uniform,
)

__all__ = [
    "BAResult",
    "ClosedCapacityResult",
    "capacity_ba",
    "capacity_binary_closed",
    "capacity_c0",
    "capacity_c1",
    "capacity_c2_closed",
    "conditional_entropy_rows",
    "mutual_info",
    "mutual_info_uniform",
]
