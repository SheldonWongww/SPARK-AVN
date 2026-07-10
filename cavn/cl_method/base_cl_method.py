import abc
import torch
import torch.nn as nn
from habitat.config import Config as CN
from habitat import logger
from typing import Dict, Any


class BaseCLMethod(abc.ABC):
    """
    Base class for continual learning methods.
    
    Provides hooks at different stages of training to allow CL methods
    to inject their logic into the DDPPO training loop.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        agent_optimizer: torch.optim.Optimizer, 
        config: CN,
        observation_space=None,
        action_space=None,
    ):
        """
        Initialize the CL method.
        
        Args:
            model: The actor-critic model
            agent_optimizer: The optimizer for the model
            config: The full configuration
            observation_space: Observation space for method-specific logic
            action_space: Action space for method-specific logic
        """
        self.model = model
        self.agent_optimizer = agent_optimizer
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.current_task_id = -1
        self.device = next(model.parameters()).device
        
    # ==================== Task-level Hooks ====================
    
    def on_task_start(self, task_id: int):
        """
        Called at the beginning of training on a new task.
        
        Args:
            task_id: ID of the new task
        """
        self.current_task_id = task_id
        logger.info(f"[{self.__class__.__name__}] Starting Task {task_id}")
    
    def on_task_end(self, task_id: int):
        """
        Called after finishing training on a task.
        Args:
            task_id: ID of the finished task
        """
        logger.info(f"[{self.__class__.__name__}] Finished Task {task_id}")
    
    # ==================== Episode-level Hooks ====================
    
    def on_episode_end(self, episode_info: Dict[str, Any]):
        """
        Called when an episode ends.
        
        Args:
            episode_info: Dictionary containing episode statistics
        """
        pass
    
    # ==================== Update-level Hooks ====================
    
    def before_collect_rollout(self, step: int):
        """
        Called before collecting a rollout step.
        Args:
            step: Current rollout step
        """
        pass
    
    def after_collect_rollout(self, rollouts, step: int):
        """
        Called after collecting a rollout step.
        Args:
            rollouts: The rollout storage
            step: Current rollout step
        """
        pass
    
    def on_rollout_complete(self, rollouts):
        """
        Called after rollout collection is complete AND returns have been computed.
        This is the ideal place for methods that need to snapshot complete trajectories.
        
        Args:
            rollouts: The rollout storage with computed returns
        """
        pass
    
    def before_update(self, rollouts, **kwargs):
        """
        Called before the model update (before agent.update()).
        Useful for replay-based methods to modify the training batch.
        
        Args:
            rollouts: The rollout storage containing training data
            **kwargs: Additional arguments
            
        Returns:
            Modified rollouts (or original rollouts if no modification)
        """
        return rollouts
    
    def calculate_cl_loss(self, rollouts, **kwargs) -> Dict[str, tuple]:
        """
        Calculate continual learning specific loss terms.
        
        Args:
            rollouts: The rollout storage
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of loss terms with their weights:
            {
                'loss_name': (loss_tensor, weight),
                ...
            }
            
        Each entry maps a loss name to a ``(loss_tensor, weight)`` tuple.
        """
        return {}
    
    def after_update(self, **kwargs):
        """
        Called after the model update.
        Useful for methods that need post-update processing.
        
        Args:
            **kwargs: Additional arguments (e.g., loss values, gradients)
        """
        pass
    
    # ==================== Gradient-level Hooks ====================
    
    def modify_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Modify gradients before the optimizer step.
        Args:
            gradients: Dictionary of parameter names to gradients
            
        Returns:
            Modified gradients
        """
        return gradients
    
    # ==================== State Management ====================
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state to be saved in checkpoints.
        
        Returns:
            Dictionary containing method-specific state
        """
        return {
            'current_task_id': self.current_task_id,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load state from a checkpoint.
        
        Args:
            state_dict: Dictionary containing method-specific state
        """
        if 'current_task_id' in state_dict:
            self.current_task_id = state_dict['current_task_id']
    
    # ==================== Metrics ====================
    
    def get_additional_metrics(self) -> Dict[str, float]:
        """
        Return method-specific metrics for logging.
        
        Returns:
            Dictionary of metric names to values
        """
        return {}
