import yaml
import torch.distributed as dist
import torch

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup(rank,world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
