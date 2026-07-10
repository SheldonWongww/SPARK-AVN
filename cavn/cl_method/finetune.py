"""
Finetune - Naive baseline without any CL mechanism
"""

import torch
from cavn.cl_method.base_cl_method import BaseCLMethod
from cavn.cl_method import cl_method_registry
from habitat.config import Config as CN
from habitat import logger


@cl_method_registry.register("finetune")
class Finetune(BaseCLMethod):
    """
    Finetune method - the simplest baseline that just continues training
    without any continual learning mechanism. This serves as a lower bound
    for continual learning performance (maximum forgetting).
    """
    
    def __init__(self, model, agent_optimizer, config, observation_space=None, action_space=None):
        super().__init__(model, agent_optimizer, config, observation_space, action_space)
        logger.info("Initializing Finetune (baseline) CL method")
        
    def on_task_start(self, task_id: int):
        """No special initialization needed for finetune."""
        super().on_task_start(task_id)
        logger.info(f"Starting finetuning on domain {task_id}")
        
    def before_update(self, rollouts, **kwargs):
        """No data modification for finetune."""
        return rollouts
    
    def calculate_cl_loss(self, rollouts, **kwargs):
        """No additional loss for finetune."""
        return {}
    
    def after_update(self, **kwargs):
        """No post-update operations for finetune."""
        pass
    
    def on_task_end(self, task_id: int):
        """No task-end operations for finetune."""
        super().on_task_end(task_id)
        logger.info(f"Completed finetuning on domain {task_id}")
