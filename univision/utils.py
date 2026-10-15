import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def available_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=False)
    return obj["model"] if isinstance(obj, dict) and "model" in obj else obj


def num_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
