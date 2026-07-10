import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from cavn.model.spark_avn.prototype_bank import (
    MAX_KMEANS_ITERS,
    PROTOTYPE_BANK_SIZE,
    merge_prototype_banks,
    routing_compatibility,
    valid_prototypes,
)


ValueEntry = Dict[str, Dict[str, Optional[torch.Tensor]]]


class AdaptiveKnowledgePool:
    def __init__(
        self,
        pool_size: int,
        prototype_bank_size: int = PROTOTYPE_BANK_SIZE,
        max_kmeans_iters: int = MAX_KMEANS_ITERS,
        merge_w_value: float = 0.5,
        merge_w_hhi: float = 0.25,
    ):
        self.pool_size = int(pool_size)
        self.prototype_bank_size = int(prototype_bank_size)
        self.max_kmeans_iters = int(max_kmeans_iters)
        self.merge_w_value = float(merge_w_value)
        self.merge_w_hhi = float(merge_w_hhi)
        self.prototype_banks: List[torch.Tensor] = []
        self.prototype_counts: List[int] = []
        self.counters: List[int] = []
        self.values: List[ValueEntry] = []

    def load(
        self,
        prototype_banks: List[torch.Tensor],
        prototype_counts: List[int],
        counters: List[int],
        values: List[ValueEntry],
    ):
        self.prototype_banks = []
        self.prototype_counts = []
        for bank, count in zip(prototype_banks, prototype_counts):
            cloned_bank, cloned_count = self._clone_bank(bank, count)
            self.prototype_banks.append(cloned_bank)
            self.prototype_counts.append(cloned_count)
        self.counters = [int(c) for c in counters]
        self.values = [self._clone_value(v) for v in values]

    def add(
        self,
        new_prototype_bank: torch.Tensor,
        new_prototype_count: int,
        new_value: ValueEntry,
        counter: int = 1,
    ):
        cloned_bank, cloned_count = self._clone_bank(new_prototype_bank, new_prototype_count)
        if cloned_count <= 0:
            raise ValueError("New slot must contain at least one valid prototype")
        self.prototype_banks.append(cloned_bank)
        self.prototype_counts.append(cloned_count)
        self.counters.append(int(counter))
        self.values.append(self._clone_value(new_value))
        while len(self.prototype_banks) > self.pool_size:
            self._merge_best_pair()

    def state(
        self,
    ) -> Tuple[List[torch.Tensor], List[int], List[int], List[ValueEntry]]:
        prototype_banks = [bank.detach().clone() for bank in self.prototype_banks]
        prototype_counts = [int(c) for c in self.prototype_counts]
        counters = [int(c) for c in self.counters]
        values = [self._clone_value(v) for v in self.values]
        return prototype_banks, prototype_counts, counters, values

    def _merge_best_pair(self):
        num_entries = len(self.prototype_banks)
        if num_entries <= 1:
            return

        idx_i, idx_j, merge_info = self._select_merge_pair()
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        count_i = self.counters[idx_i]
        count_j = self.counters[idx_j]
        merged_summary = merge_prototype_banks(
            self.prototype_banks[idx_i],
            self.prototype_counts[idx_i],
            self.prototype_banks[idx_j],
            self.prototype_counts[idx_j],
            bank_size=self.prototype_bank_size,
            max_iters=self.max_kmeans_iters,
        )
        merged_value = self._merge_values(
            self.values[idx_i],
            self.values[idx_j],
            float(count_i),
            float(count_j),
        )

        logging.info(
            "[SPARK-AVN] merge pair (%d,%d): counters=(%d,%d), proto=%.4f, value=%.4f, hhi=%.4f, total=%.4f, prototypes=(%d,%d)->%d",
            idx_i,
            idx_j,
            count_i,
            count_j,
            merge_info["proto_score"],
            merge_info["value_score"],
            merge_info["hhi_penalty"],
            merge_info["total_score"],
            self.prototype_counts[idx_i],
            self.prototype_counts[idx_j],
            merged_summary.prototype_count,
        )

        keep_indices = [idx for idx in range(num_entries) if idx not in (idx_i, idx_j)]
        self.prototype_banks = [self.prototype_banks[idx] for idx in keep_indices] + [
            merged_summary.prototype_bank.detach().clone()
        ]
        self.prototype_counts = [self.prototype_counts[idx] for idx in keep_indices] + [
            int(merged_summary.prototype_count)
        ]
        self.counters = [self.counters[idx] for idx in keep_indices] + [count_i + count_j]
        self.values = [self.values[idx] for idx in keep_indices] + [merged_value]

    def _select_merge_pair(self) -> Tuple[int, int, Dict[str, float]]:
        num_entries = len(self.prototype_banks)
        if num_entries <= 1:
            return 0, 0, {
                "proto_score": 0.0,
                "value_score": 0.0,
                "hhi_penalty": 0.0,
                "total_score": 0.0,
            }

        total_counter = float(sum(self.counters))
        best_pair = None
        best_info = None

        for idx_i in range(num_entries):
            for idx_j in range(idx_i + 1, num_entries):
                proto_raw = routing_compatibility(
                    self.prototype_banks[idx_i],
                    self.prototype_counts[idx_i],
                    self.prototype_banks[idx_j],
                    self.prototype_counts[idx_j],
                )
                value_raw = self._compute_value_cosine(
                    self.values[idx_i],
                    self.values[idx_j],
                )

                proto_score = 0.5 * (proto_raw + 1.0)
                value_score = 0.5 * (value_raw + 1.0)
                p_i = float(self.counters[idx_i]) / max(total_counter, 1e-12)
                p_j = float(self.counters[idx_j]) / max(total_counter, 1e-12)
                hhi_penalty = math.sqrt(max(4.0 * p_i * p_j, 0.0))
                total_score = (
                    proto_score
                    + self.merge_w_value * value_score
                    - self.merge_w_hhi * hhi_penalty
                )

                info = {
                    "proto_score": float(proto_score),
                    "value_score": float(value_score),
                    "hhi_penalty": float(hhi_penalty),
                    "total_score": float(total_score),
                }
                if best_info is None or total_score > best_info["total_score"]:
                    best_pair = (idx_i, idx_j)
                    best_info = info

        assert best_pair is not None and best_info is not None
        return best_pair[0], best_pair[1], best_info

    @staticmethod
    def _compute_value_cosine(value_i: ValueEntry, value_j: ValueEntry) -> float:
        dot_sum = 0.0
        norm_i = 0.0
        norm_j = 0.0

        for layer_name in sorted(value_i.keys()):
            wi = value_i[layer_name]["weight"]
            wj = value_j[layer_name]["weight"]
            if wi is not None and wj is not None:
                wi = wi.reshape(-1)
                wj = wj.reshape(-1)
                dot_sum += float(torch.dot(wi, wj).item())
                norm_i += float(torch.dot(wi, wi).item())
                norm_j += float(torch.dot(wj, wj).item())

            bi = value_i[layer_name]["bias"]
            bj = value_j[layer_name]["bias"]
            if bi is not None and bj is not None:
                bi = bi.reshape(-1)
                bj = bj.reshape(-1)
                dot_sum += float(torch.dot(bi, bj).item())
                norm_i += float(torch.dot(bi, bi).item())
                norm_j += float(torch.dot(bj, bj).item())

        if norm_i <= 1e-12 or norm_j <= 1e-12:
            return 0.0
        return dot_sum / math.sqrt(norm_i * norm_j)

    def _clone_bank(self, bank: torch.Tensor, count: int) -> Tuple[torch.Tensor, int]:
        if bank.dim() != 2:
            raise ValueError("Prototype bank must be a 2D tensor")
        valid = valid_prototypes(bank, count)
        if valid.size(0) > self.prototype_bank_size:
            valid = valid[: self.prototype_bank_size]
        cloned = bank.detach().clone().new_zeros((self.prototype_bank_size, bank.size(-1)))
        if valid.numel() > 0:
            cloned[: valid.size(0)] = valid
        return cloned, int(valid.size(0))

    @staticmethod
    def _clone_value(value: ValueEntry) -> ValueEntry:
        result: ValueEntry = {}
        for layer_name, layer_value in value.items():
            weight = layer_value["weight"]
            bias = layer_value["bias"]
            cloned_weight = None if weight is None else weight.detach().clone()
            cloned_bias = None if bias is None else bias.detach().clone()
            result[layer_name] = {"weight": cloned_weight, "bias": cloned_bias}
        return result

    def _merge_values(
        self,
        value_i: ValueEntry,
        value_j: ValueEntry,
        weight_i: float,
        weight_j: float,
    ) -> ValueEntry:
        merged: ValueEntry = {}
        for layer_name in value_i.keys():
            wi = value_i[layer_name]["weight"]
            wj = value_j[layer_name]["weight"]
            bi = value_i[layer_name]["bias"]
            bj = value_j[layer_name]["bias"]

            merged[layer_name] = {
                "weight": self._merge_tensor(wi, wj, weight_i, weight_j),
                "bias": self._merge_tensor(bi, bj, weight_i, weight_j),
            }
        return merged

    @staticmethod
    def _merge_tensor(
        tensor_i: Optional[torch.Tensor],
        tensor_j: Optional[torch.Tensor],
        weight_i: float,
        weight_j: float,
    ) -> Optional[torch.Tensor]:
        if tensor_i is None and tensor_j is None:
            return None
        if tensor_i is None:
            return tensor_j.detach().clone()
        if tensor_j is None:
            return tensor_i.detach().clone()

        denom = max(weight_i + weight_j, 1e-12)
        return ((weight_i * tensor_i) + (weight_j * tensor_j)) / denom
