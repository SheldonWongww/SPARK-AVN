from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


PROTOTYPE_BANK_SIZE = 5
MAX_KMEANS_ITERS = 10


@dataclass
class TaskRoutingSummary:
    prototype_bank: torch.Tensor
    prototype_count: int

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "TaskRoutingSummary":
        bank = self.prototype_bank
        if device is not None or dtype is not None:
            bank = bank.to(device=device or bank.device, dtype=dtype or bank.dtype)
        return TaskRoutingSummary(
            prototype_bank=bank,
            prototype_count=int(self.prototype_count),
        )


def safe_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    return F.normalize(tensor, p=2, dim=dim, eps=1e-12)


def build_task_routing_summary(
    candidates: torch.Tensor,
    bank_size: int = PROTOTYPE_BANK_SIZE,
    max_iters: int = MAX_KMEANS_ITERS,
) -> TaskRoutingSummary:
    bank, count = compress_prototype_bank(
        candidates,
        bank_size=bank_size,
        max_iters=max_iters,
    )
    return TaskRoutingSummary(prototype_bank=bank, prototype_count=count)


def merge_prototype_banks(
    bank_i: torch.Tensor,
    count_i: int,
    bank_j: torch.Tensor,
    count_j: int,
    bank_size: int = PROTOTYPE_BANK_SIZE,
    max_iters: int = MAX_KMEANS_ITERS,
) -> TaskRoutingSummary:
    valid_i = valid_prototypes(bank_i, count_i)
    valid_j = valid_prototypes(bank_j, count_j)
    if valid_i.numel() == 0 or valid_j.numel() == 0:
        raise ValueError("Cannot merge empty prototype banks")

    candidates = torch.cat([valid_i, valid_j], dim=0)
    bank, count = compress_prototype_bank(
        candidates,
        bank_size=bank_size,
        max_iters=max_iters,
    )
    return TaskRoutingSummary(prototype_bank=bank, prototype_count=count)


def routing_compatibility(
    bank_i: torch.Tensor,
    count_i: int,
    bank_j: torch.Tensor,
    count_j: int,
) -> float:
    valid_i = valid_prototypes(bank_i, count_i)
    valid_j = valid_prototypes(bank_j, count_j)
    if valid_i.numel() == 0 or valid_j.numel() == 0:
        return float("-inf")

    sims = torch.matmul(valid_i, valid_j.t())
    a_to_b = sims.max(dim=1).values.mean()
    b_to_a = sims.max(dim=0).values.mean()
    return float((0.5 * (a_to_b + b_to_a)).item())


def valid_prototypes(bank: torch.Tensor, count: int) -> torch.Tensor:
    count = max(0, min(int(count), int(bank.size(0))))
    if count <= 0:
        return bank.new_zeros((0, bank.size(-1)))
    return safe_normalize(bank[:count], dim=-1)


def select_medoid(candidates: torch.Tensor) -> torch.Tensor:
    candidates = _prepare_candidates(candidates)
    sims = torch.matmul(candidates, candidates.t())
    medoid_idx = int(torch.argmax(sims.mean(dim=1)).item())
    return candidates[medoid_idx]


def compress_prototype_bank(
    candidates: torch.Tensor,
    bank_size: int = PROTOTYPE_BANK_SIZE,
    max_iters: int = MAX_KMEANS_ITERS,
) -> Tuple[torch.Tensor, int]:
    candidates = _prepare_candidates(candidates)
    if candidates.size(0) == 0:
        raise ValueError("Prototype candidates must be non-empty")

    bank_size = max(1, int(bank_size))
    max_iters = max(1, int(max_iters))
    count = min(bank_size, int(candidates.size(0)))

    if candidates.size(0) <= count:
        bank = candidates.new_zeros((bank_size, candidates.size(-1)))
        bank[:count] = candidates[:count]
        return bank, count

    centers = _initialize_kmeans_centers(candidates, count)
    assignments = None

    for _ in range(max_iters):
        sims = torch.matmul(candidates, centers.t())
        new_assignments = torch.argmax(sims, dim=1)
        if assignments is not None and torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        centers = _update_kmeans_centers(candidates, assignments, centers)

    bank = candidates.new_zeros((bank_size, candidates.size(-1)))
    bank[:count] = centers
    return bank, count


def _prepare_candidates(candidates: torch.Tensor) -> torch.Tensor:
    if candidates is None:
        raise ValueError("Prototype candidates must not be None")
    if candidates.dim() == 1:
        candidates = candidates.unsqueeze(0)
    if candidates.dim() != 2:
        raise ValueError("Prototype candidates must be a 2D tensor")
    return safe_normalize(candidates.detach().clone(), dim=-1)


def _initialize_kmeans_centers(candidates: torch.Tensor, count: int) -> torch.Tensor:
    selected = [select_medoid(candidates)]

    while len(selected) < count:
        selected_tensor = torch.stack(selected, dim=0)
        sims = torch.matmul(candidates, selected_tensor.t())
        max_sim = sims.max(dim=1).values
        next_idx = int(torch.argmin(max_sim).item())
        selected.append(candidates[next_idx])

    return torch.stack(selected, dim=0)


def _update_kmeans_centers(
    candidates: torch.Tensor,
    assignments: torch.Tensor,
    prev_centers: torch.Tensor,
) -> torch.Tensor:
    new_centers = prev_centers.clone()
    for cluster_idx in range(prev_centers.size(0)):
        members = candidates[assignments == cluster_idx]
        if members.numel() == 0:
            continue
        center = members.mean(dim=0)
        new_centers[cluster_idx] = safe_normalize(center, dim=-1)
    return new_centers
