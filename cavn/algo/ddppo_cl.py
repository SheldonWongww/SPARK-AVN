import torch
import torch.nn as nn
from typing import Dict, Tuple
from ss_baselines.savi.ddppo.algo.ddppo import DDPPO


class DDPPO_CL(DDPPO):
    """DDPPO with continual-learning hooks during PPO updates."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_tracker = {}
        self.cl_method = None
        self.current_task_id = -1
    
    def set_cl_method(self, cl_method):
        """Set the CL method for this agent."""
        self.cl_method = cl_method
    
    def set_task_id(self, task_id: int):
        """Set the current task id."""
        self.current_task_id = task_id
    
    def update_with_cl(
        self, 
        rollouts,
        task_id: int = None,
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Modified update function that includes continual learning losses.
        
        Args:
            rollouts: Experience buffer with computed returns.
            task_id: Current domain id.
        """
        if task_id is not None:
            self.current_task_id = task_id
        
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if self.use_normalized_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5
            )

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        cl_loss_epoch = {}

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    ext_memory_batch,
                    ext_memory_masks_batch,
                ) = sample

                eval_result = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                    ext_memory_batch,
                    ext_memory_masks_batch,
                )

                # Support both standard and extended policy return formats.
                if len(eval_result) == 6:
                    values, action_log_probs, dist_entropy, _, _, _ = eval_result
                elif len(eval_result) == 5:
                    values, action_log_probs, dist_entropy, _, _ = eval_result
                else:
                    raise ValueError(
                        f"Unexpected evaluate_actions return length: {len(eval_result)}"
                    )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # ==================== Combine losses ====================
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )
                
                # ==================== CL Loss ====================
                if self.cl_method is not None:
                    step_cl_losses = self.cl_method.calculate_cl_loss(None)
                    
                    for loss_name, (loss_tensor, weight) in step_cl_losses.items():
                        if loss_tensor is not None:
                            un_key = f"{loss_name}_unweighted"
                            if un_key not in cl_loss_epoch:
                                cl_loss_epoch[un_key] = 0.0
                            cl_loss_epoch[un_key] += float(loss_tensor.detach().item())
                        
                        if loss_tensor is not None and weight > 0:
                            weighted_cl_loss = loss_tensor * weight
                            total_loss = total_loss + weighted_cl_loss
                            
                            if loss_name not in cl_loss_epoch:
                                cl_loss_epoch[loss_name] = 0.0
                            cl_loss_epoch[loss_name] += weighted_cl_loss.item()

                # ==================== Backward pass ====================
                self.optimizer.zero_grad()
                total_loss.backward()
                
                if self.cl_method is not None and hasattr(self.cl_method, 'modify_gradients'):
                    gradients = {}
                    for name, param in self.actor_critic.named_parameters():
                        if param.grad is not None:
                            gradients[name] = param.grad.clone()
                    
                    modified_gradients = self.cl_method.modify_gradients(gradients)
                    
                    for name, param in self.actor_critic.named_parameters():
                        if name in modified_gradients:
                            param.grad.copy_(modified_gradients[name])
                
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        # Average over all updates
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        
        for loss_name in cl_loss_epoch:
            cl_loss_epoch[loss_name] /= num_updates

        return (
            value_loss_epoch, 
            action_loss_epoch, 
            dist_entropy_epoch,
            cl_loss_epoch
        )
    
    def get_loss_tracker(self) -> Dict[str, float]:
        """Get tracked loss values"""
        return self._loss_tracker.copy()
    
    def reset_loss_tracker(self):
        """Reset loss tracker"""
        self._loss_tracker.clear()
