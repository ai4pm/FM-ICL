from __future__ import annotations

import os
import copy
import socket
import numpy as np 
import random as rn

from typing import List, Optional, Union, Tuple

from dataclasses import dataclass, field
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from datetime import timedelta

@dataclass
class HParams:
    batch_size: int = 20
    learning_rate: float = 1e-2
    lr_decay: float = 0.0          # per-epoch exponential decay (0 = off)
    dropout: float = 0.5
    n_epochs: int = 1
    momentum: float = 0.9
    L1_reg: float = 0.0
    L2_reg: float = 1e-3           # via optimizer weight_decay
    hiddenLayers: List[int] = field(default_factory=lambda: [128, 64, 32, 16])
    cnn_kernel_size: int = 3
    seed: Optional[int] = 42
    balance_probabilities: bool = False

def develop_multimodal_base_models(
    X_multimodal_train: List[np.ndarray],
    y_multimodal_train: List[np.ndarray],
    models_to_load: List[Optional[nn.Module]], 
    h: Optional[HParams] = None,
) -> List[nn.Module]:    
    
    developed_models = [] 
    for X_train, y_train, pretrained_model in zip(X_multimodal_train, y_multimodal_train, models_to_load):
        model, _ = train_binary_mlp(
            X_train=X_train, y_train=y_train,
            h=h,
            model = pretrained_model, 
            # we can use a small held-out split from the source-train as validation
            val_split=0.33,
        )
        developed_models.append(model)

    return developed_models

def _compute_bce_loss(model: nn.Module, dl: DataLoader, device, loss_fn, l1_reg: float = 0.0):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb).squeeze(1)
            loss = loss_fn(logit, yb)
            if l1_reg and l1_reg > 0:
                l1 = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_reg * l1
            bs = yb.shape[0]
            total += loss.item() * bs
            n += bs
    return total / max(1, n)

class FeatureCNN1D(nn.Module):
    def __init__(self, in_dim: int, h: HParams):
        super().__init__()

        layers = []
        current_length = int(in_dim)
        target_lengths = [128, 64, 32, 16]
        for target_length in target_lengths:
            layers += [
                nn.Conv1d(1, 1, kernel_size=3, padding=1),
                nn.BatchNorm1d(1),
                nn.ReLU(inplace=True),
            ]
            if current_length > target_length:
                stride = max(1, current_length // target_length)
                kernel_size = current_length - (target_length - 1) * stride
                layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
                current_length = target_length

        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(current_length, 1)

    def forward(self, x):
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim > 3:
            x = x.view(x.shape[0], 1, -1)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.output(x)


# Kept under the old name because the training code imports this builder.
def build_mlp(in_dim: int, h: HParams) -> nn.Module:
    return FeatureCNN1D(in_dim, h)


def train_binary_mlp(
    X_train: Union[torch.Tensor, "np.ndarray"],
    y_train: Union[torch.Tensor, "np.ndarray"],   # float 0/1
    *,
    h: Optional[HParams] = None,
    model: Optional[nn.Module] = None,
    X_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    y_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    val_split: float = 0.33,
) -> Tuple[nn.Module, float]:
    """
    Train the default TL 1D CNN and SAVE the best model
    (lowest val loss; if no val, lowest train loss).
    Returns: (model_loaded_with_best_state, best_metric_value)
    """
    return _train_binary_model(
        X_train,
        y_train,
        h=h,
        model=model,
        X_val=X_val,
        y_val=y_val,
        val_split=val_split,
        model_builder=build_mlp,
    )


def _train_binary_model(
    X_train: Union[torch.Tensor, "np.ndarray"],
    y_train: Union[torch.Tensor, "np.ndarray"],   # float 0/1
    *,
    h: Optional[HParams] = None,
    model: Optional[nn.Module] = None,
    X_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    y_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    val_split: float = 0.33,
    model_builder=build_mlp,
) -> Tuple[nn.Module, float]:
    """
    Train the default TL 1D CNN and SAVE the best model
    (lowest val loss; if no val, lowest train loss).
    Returns: (model_loaded_with_best_state, best_metric_value)
    """
    if h is None:
        h = HParams()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeds
    if h.seed is not None:
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(h.seed)
        np.random.seed(h.seed)
        rn.seed(h.seed)

    # Tensors
    if not torch.is_tensor(X_train): X_train = torch.as_tensor(X_train)
    if not torch.is_tensor(y_train): y_train = torch.as_tensor(y_train)
    if X_val is not None and not torch.is_tensor(X_val): X_val = torch.as_tensor(X_val)
    if y_val is not None and not torch.is_tensor(y_val): y_val = torch.as_tensor(y_val)
    X_train = X_train.float(); y_train = y_train.float().view(-1)
    if X_val is not None: X_val = X_val.float()
    if y_val is not None: y_val = y_val.float().view(-1)

    # Model
    in_dim = X_train.shape[1]
    if model == None: 
        model = model_builder(in_dim, h)
    model.to(device)


    # Data
    full_ds = TensorDataset(X_train, y_train)
    if X_val is not None and y_val is not None:
        train_dl = DataLoader(full_ds, batch_size=h.batch_size, shuffle=True)
        val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=h.batch_size, shuffle=False)
    else:
        n_total = len(full_ds)
        n_val = max(0, int(round(val_split * n_total))) if n_total > 1 else 0
        n_train = max(0, n_total - n_val)
        if n_val > 0 and n_train > 0:
            tr_ds, va_ds = random_split(full_ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(h.seed or 0))
            train_dl = DataLoader(tr_ds, batch_size=h.batch_size, shuffle=True)
            val_dl = DataLoader(va_ds, batch_size=h.batch_size, shuffle=False)
        else:
            train_dl = DataLoader(full_ds, batch_size=h.batch_size, shuffle=True)
            val_dl = None

    # Optimizer / scheduler
    opt = torch.optim.SGD(model.parameters(), lr=h.learning_rate,
                          momentum=h.momentum, weight_decay=h.L2_reg)
    sched = None
    if h.lr_decay and h.lr_decay > 0.0:
        gamma = max(1e-8, 1.0 - h.lr_decay)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    loss_fn = nn.BCEWithLogitsLoss()

    # Train
    best_metric = float("inf")
    best_state = None
    # epoch_iter = trange(h.n_epochs, desc="Training", leave=True)
    for _ in range(h.n_epochs):
        # pdb.set_trace()
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb).squeeze(1)
            loss = loss_fn(logit, yb)
            if h.L1_reg and h.L1_reg > 0:
                l1 = sum(p.abs().sum() for p in model.parameters())
                loss = loss + h.L1_reg * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = yb.shape[0]
            running_loss += loss.item() * bs
            n_seen += bs
        if sched is not None:
            sched.step()

        train_loss = running_loss / max(1, n_seen)
        if val_dl is not None:
            val_loss = _compute_bce_loss(model, val_dl, device, loss_fn)
            metric = val_loss
        else:
            val_loss = None
            metric = train_loss

        # save best
        if metric < best_metric:
            best_metric = metric
            # best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_state = copy.deepcopy(model.state_dict())
            # torch.save(best_state, best_model_path)

        # epoch_iter.set_postfix({
        #     "train_loss": f"{train_loss:.4f}",
        #     "val_loss": "N/A" if val_loss is None else f"{val_loss:.4f}",
        #     "best": f"{best_metric:.4f}",
        # })

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metric

# helper: logits -> [N, 2] proba for compute_metrics(...)
def _probs_from_logits_binary(logits_np: np.ndarray) -> np.ndarray:
    # p1 = 1.0 / (1.0 + np.exp(-logits_np))
    p1 = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -500, 500)))
    return np.stack([1.0 - p1, p1], axis=1)

# ======================================
# The module of the distributed training  
# ======================================


# -------------------------
# DDP setup/cleanup for mp.spawn
# -------------------------
def _ddp_setup(rank: int, world_size: int, *, master_addr: str = "127.0.0.1", master_port: str = "29501"):
    """
    Setup process group for mp.spawn.

    NOTE: You must ensure MASTER_PORT is free; change master_port if you see EADDRINUSE.
    """
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=120))


def _ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _broadcast_object(obj, *, src: int = 0):
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# -------------------------
# mp.spawn worker (per-rank)
# -------------------------
def _ddp_worker_train(
    rank: int,
    world_size: int,
    X_train: Union[torch.Tensor, "np.ndarray"],
    y_train: Union[torch.Tensor, "np.ndarray"],
    h,
    model: Optional[nn.Module],
    X_val: Optional[Union[torch.Tensor, "np.ndarray"]],
    y_val: Optional[Union[torch.Tensor, "np.ndarray"]],
    val_split: float,
    return_dict,
    master_addr: str,
    master_port: str,
) -> None:
    """
    mp.spawn worker: runs once per rank.
    Writes return_dict[0] = {"best_metric": float, "best_state": state_dict} from rank 0.
    """
    _ddp_setup(rank, world_size, master_addr=master_addr, master_port=master_port)
    try:
        # ---- hyper-parameters ----
        if h is None:
            h = HParams()

        # ---- device ----
        if torch.cuda.is_available():
            # With mp.spawn on a single node, rank typically matches GPU id.
            # If you ever do multi-node or custom mapping, pass local_rank separately.
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")

        # ---- Seeds (per-rank but deterministic) ----
        if getattr(h, "seed", None) is not None:
            seed = int(h.seed) + int(rank)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            rn.seed(seed)

        # ---- Tensors (keep on CPU; let DataLoader move minibatches) ----
        if not torch.is_tensor(X_train):
            X_train = torch.as_tensor(X_train)
        if not torch.is_tensor(y_train):
            y_train = torch.as_tensor(y_train)
        if X_val is not None and not torch.is_tensor(X_val):
            X_val = torch.as_tensor(X_val)
        if y_val is not None and not torch.is_tensor(y_val):
            y_val = torch.as_tensor(y_val)

        X_train = X_train.float()
        y_train = y_train.float().view(-1)
        if X_val is not None:
            X_val = X_val.float()
        if y_val is not None:
            y_val = y_val.float().view(-1)

        # ---- Model ----
        in_dim = int(X_train.shape[1])
        if model is None:
            model = build_mlp(in_dim, h)
        model.to(device)

        # ---- Wrap with DDP ----
        if torch.cuda.is_available():
            ddp_model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
            )
        else:
            ddp_model = DDP(model, find_unused_parameters=False)

        # ---- Data ----
        full_ds = TensorDataset(X_train, y_train)
        if X_val is not None and y_val is not None:
            train_ds = full_ds
            val_ds = TensorDataset(X_val, y_val)
        else:
            n_total = len(full_ds)
            n_val = max(0, int(round(val_split * n_total))) if n_total > 1 else 0
            n_train = max(0, n_total - n_val)
            if n_val > 0 and n_train > 0:
                tr_ds, va_ds = random_split(
                    full_ds,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(getattr(h, "seed", None) or 0),
                )
                train_ds, val_ds = tr_ds, va_ds
            else:
                train_ds, val_ds = full_ds, None

        # ---- Samplers ----
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=h.batch_size,
            sampler=train_sampler,
            shuffle=False,
        )

        if val_ds is not None:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            val_dl = DataLoader(
                val_ds,
                batch_size=h.batch_size,
                sampler=val_sampler,
                shuffle=False,
            )
        else:
            val_dl = None

        # ---- Optimizer / scheduler ----
        opt = torch.optim.SGD(
            ddp_model.parameters(),
            lr=h.learning_rate,
            momentum=h.momentum,
            weight_decay=h.L2_reg,
        )
        sched = None
        if getattr(h, "lr_decay", None) and h.lr_decay > 0.0:
            gamma = max(1e-8, 1.0 - h.lr_decay)
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

        loss_fn = nn.BCEWithLogitsLoss()

        best_metric = float("inf")
        best_state = None

        for epoch in range(h.n_epochs):
            ddp_model.train()
            train_sampler.set_epoch(epoch)

            running_loss = 0.0
            n_seen = 0

            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logit = ddp_model(xb).squeeze(1)
                loss = loss_fn(logit, yb)

                if getattr(h, "L1_reg", None) and h.L1_reg > 0:
                    l1 = sum(p.abs().sum() for p in ddp_model.module.parameters())
                    loss = loss + h.L1_reg * l1

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = yb.shape[0]
                running_loss += loss.detach().item() * bs
                n_seen += bs

            if sched is not None:
                sched.step()

            # ---- Aggregate train loss across ranks ----
            train_sum = torch.tensor([running_loss, n_seen], device=device, dtype=torch.float64)
            dist.all_reduce(train_sum, op=dist.ReduceOp.SUM)
            global_train_loss = (train_sum[0] / torch.clamp(train_sum[1], min=1.0)).item()

            if val_dl is not None:
                ddp_model.eval()
                with torch.no_grad():
                    val_running = 0.0
                    val_seen = 0
                    for xb, yb in val_dl:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        logit = ddp_model(xb).squeeze(1)
                        vloss = loss_fn(logit, yb)
                        bs = yb.shape[0]
                        val_running += vloss.detach().item() * bs
                        val_seen += bs

                val_sum = torch.tensor([val_running, val_seen], device=device, dtype=torch.float64)
                dist.all_reduce(val_sum, op=dist.ReduceOp.SUM)
                global_val_loss = (val_sum[0] / torch.clamp(val_sum[1], min=1.0)).item()
                metric = global_val_loss
            else:
                metric = global_train_loss

            # ---- Save best on rank 0 only ----
            if _is_main_process(rank) and metric < best_metric:
                best_metric = float(metric)
                best_state = {k: v.detach().cpu().clone() for k, v in ddp_model.module.state_dict().items()}

        # ---- Broadcast best payload so every rank *could* load it (optional) ----
        if _is_main_process(rank):
            payload = {"best_metric": best_metric, "best_state": best_state}
        else:
            payload = None

        payload = _broadcast_object(payload, src=0)

        # If you want all ranks to end with the same best weights loaded:
        ddp_model.module.load_state_dict(payload["best_state"])

        # Only rank 0 writes back to the parent
        if _is_main_process(rank):
            return_dict[0] = payload

    finally:
        print(f"[rank {rank}] entering cleanup", flush=True)
        _ddp_cleanup()
        print(f"[rank {rank}] finished cleanup", flush=True)

def pick_free_port() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)

# -------------------------
# Public entrypoint (mp.spawn)
# -------------------------
def ddp_train_nn(
    X_train: Union[torch.Tensor, "np.ndarray"],
    y_train: Union[torch.Tensor, "np.ndarray"],
    *,
    h: Optional[HParams] = None,
    model: Optional[nn.Module] = None,
    X_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    y_val: Optional[Union[torch.Tensor, "np.ndarray"]] = None,
    val_split: float = 0.33,
    world_size: Optional[int] = None,
) -> Tuple[nn.Module, float]:
    """
    mp.spawn entrypoint. Spawns one process per GPU (or per requested world_size on CPU).
    Returns (model_loaded_with_best_state, best_metric) to the parent process.
    """


    master_addr = "127.0.0.1"
    master_port = pick_free_port()

    if h is None:
        h = HParams()

    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2

    if world_size < 2:
        # fall back to your single-process trainer
        return train_binary_mlp(X_train, y_train, h=h, model=model, X_val=X_val, y_val=y_val, val_split=val_split)

    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _ddp_worker_train,
        args=(world_size, X_train, y_train, h, model, X_val, y_val, val_split, return_dict, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )

    payload = return_dict[0]
    best_metric = float(payload["best_metric"])
    best_state = payload["best_state"]

    # Recreate model in parent and load best weights
    if not torch.is_tensor(X_train):
        X_train = torch.as_tensor(X_train)
    in_dim = int(X_train.shape[1])
    if model is None:
        model = build_mlp(in_dim, h)
    model.load_state_dict(best_state)

    return model, best_metric
