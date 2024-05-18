# Monkey patch the `build_metadata` method
# Refer to https://github.com/b-chu/pytorch/blob/3ccc8f3357447f83ce56dd9d7618e520e4a11e97/torch/distributed/_shard/sharding_spec/chunk_sharding_spec.py
import torch
from dataclasses import dataclass
import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device
)
from torch.distributed._shard._utils import narrow_tensor
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from typing import List, Union, TYPE_CHECKING
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)

from torch.distributed._shard.sharding_spec.api import ShardingSpec

if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec


def patched_build_metadata(self,
                           tensor_sizes: torch.Size,
                           tensor_properties: sharded_tensor_meta.TensorProperties,
                           ) -> sharded_tensor_meta.ShardedTensorMetadata:
    tensor_num_dim = len(tensor_sizes)

    self._verify_dim(self.dim)
    if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
        raise ValueError(f"Invalid sharding dim: {self.dim}")

    shards_metadata = []
    sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
    chunks = len(self.placements)
    split_size = get_split_size(sharding_dim_size, chunks)
    for idx, placement in enumerate(self.placements):
        # generate ShardMetadata for each placement device
        chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        shard_size = list(tensor_sizes)
        current_offsets = [0] * tensor_num_dim
        current_offsets[self.dim] = split_size * idx  # type: ignore[index]
        shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

        shard_metadata = ShardMetadata(
            shard_offsets=current_offsets,
            shard_sizes=shard_size,
            placement=placement,
        )
        shards_metadata.append(shard_metadata)

    return sharded_tensor_meta.ShardedTensorMetadata(
        shards_metadata,
        tensor_sizes,
        tensor_properties
    )

ChunkShardingSpec.build_metadata = patched_build_metadata
print(f"ChunkShardingSpec.build_metadata has been patched...")