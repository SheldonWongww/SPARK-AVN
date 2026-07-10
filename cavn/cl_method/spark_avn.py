from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from habitat import logger
from habitat.config import Config as CN

from cavn.cl_method.base_cl_method import BaseCLMethod
from cavn.cl_method import cl_method_registry


@cl_method_registry.register("spark_avn")
class SparkAVN(BaseCLMethod):
    def __init__(
        self,
        model: nn.Module,
        agent_optimizer: torch.optim.Optimizer,
        config: CN,
        observation_space=None,
        action_space=None,
    ):
        super().__init__(model, agent_optimizer, config, observation_space, action_space)
        self.centroid_buffer: List[torch.Tensor] = []
        self.training_history: List[int] = []
        logger.info("[SPARK-AVN] Initialized")

    def on_task_start(self, task_id: int):
        super().on_task_start(task_id)
        self.centroid_buffer = []
        if task_id not in self.training_history:
            self.training_history.append(task_id)

    def add_local_centroid(self, centroid: Optional[torch.Tensor]):
        if centroid is None:
            return
        self.centroid_buffer.append(centroid.detach().clone().cpu())

    def drain_centroid_buffer(self) -> Optional[torch.Tensor]:
        if len(self.centroid_buffer) == 0:
            self.centroid_buffer = []
            return None
        global_centroid = torch.stack(self.centroid_buffer, dim=0)
        self.centroid_buffer = []
        return global_centroid

    def on_task_end(self, task_id: int):
        super().on_task_end(task_id)
        self.centroid_buffer = []

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "training_history": self.training_history,
            }
        )
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.training_history = state_dict.get("training_history", [])

    def get_additional_metrics(self) -> Dict[str, float]:
        return {
            "spark_avn_num_tasks": float(len(self.training_history)),
            "spark_avn_pool_active": float(
                self.model.active_pool_count if hasattr(self.model, "active_pool_count") else 0
            ),
            "spark_avn_buffered_updates": float(len(self.centroid_buffer)),
        }
