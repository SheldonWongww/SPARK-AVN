#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pdb
from collections import defaultdict

import torch


class ExternalMemory:
    """外部记忆类，用于跟踪观察值随时间的变化
    
    这是一个循环缓冲区实现的外部记忆系统，用于存储和管理历史观察值。
    支持多个并行环境，每个环境都有独立的记忆空间。
    
    主要功能：
    1. 存储历史观察值特征
    2. 管理记忆的有效性mask
    3. 处理容量溢出（FIFO策略）
    4. 支持episode结束时的记忆重置
    """
    
    def __init__(self, num_envs, total_size, capacity, dim, num_copies=1):
        """初始化外部记忆
        
        Args:
            num_envs (int): 并行环境数量
            total_size (int): 记忆总大小（容量 + 额外缓冲区大小用于rollout更新）
            capacity (int): 每个episode的记忆容量
            dim (int): 观察值特征维度
            num_copies (int): 为高效训练维护的数据副本数量，默认为1
        """
        self.total_size = total_size
        self.capacity = capacity
        self.dim = dim
        
        # 初始化记忆mask，标记哪些位置是有效的
        # 形状：(环境数, 总大小)
        self.masks = torch.zeros(num_envs, self.total_size)
        
        # 初始化记忆存储
        # 形状：(总大小, 副本数, 环境数, 特征维度)
        self.memory = torch.zeros(self.total_size, num_copies, num_envs, self.dim)
        
        # 当前写入索引（循环使用）
        self.idx = 0

    def insert(self, em_features, not_done_masks):
        """插入新的外部记忆特征
        
        Args:
            em_features (torch.Tensor): 要插入的外部记忆特征,(num_copies, num_envs, dim)
            not_done_masks (torch.Tensor): 标记episode是否未结束的mask,(num_envs, 1)
        """
        # 更新记忆存储，添加新的记忆作为有效条目
        self.memory[self.idx].copy_(em_features.unsqueeze(0))
        
        # 处理容量溢出：如果某个环境的记忆已满，移除最旧的条目
        capacity_overflow_flag = self.masks.sum(1) == self.capacity
        assert(not torch.any(self.masks.sum(1) > self.capacity))  # 确保不超过容量
        self.masks[capacity_overflow_flag, self.idx - self.capacity] = 0.0  # 把最老的那条记忆标记为无效
        
        # 标记当前位置为有效
        self.masks[:, self.idx] = 1.0
        
        # 如果episode结束，将整个记忆mask清零
        self.masks *= not_done_masks
        
        # 更新索引（循环使用）
        self.idx = (self.idx + 1) % self.total_size

    def pop_at(self, idx):
        """在指定索引处弹出记忆条目
        
        Args:
            idx (int): 要弹出的索引位置
        """
        # 从mask中移除指定索引
        self.masks = torch.cat([self.masks[:idx, :], self.masks[idx+1:, :]], dim=0)
        # 从记忆中移除指定索引
        self.memory = torch.cat([self.memory[:, :, :idx, :], self.memory[:, :, idx+1:, :]], dim=2)

    def to(self, device):
        """将所有张量移动到指定设备
        
        Args:
            device: 目标设备
        """
        self.masks = self.masks.to(device)
        self.memory = self.memory.to(device)


class RolloutStorage:
    r"""强化学习训练器的rollout信息存储类
    
    该类用于存储强化学习训练过程中的各种数据，包括观察值、动作、奖励、价值预测等。
    支持循环神经网络(RNN)的隐藏状态存储，以及外部记忆机制。
    
    主要功能：
    1. 存储环境交互数据（观察值、动作、奖励等）
    2. 管理循环神经网络的隐藏状态
    3. 支持外部记忆机制用于存储历史信息
    4. 提供数据批处理功能用于训练
    5. 计算回报值（returns）和优势值（advantages）
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        use_external_memory,
        external_memory_size,
        external_memory_capacity,
        external_memory_dim,
        num_recurrent_layers=1,
    ):
        """初始化RolloutStorage存储结构
        
        Args:
            num_steps (int): 每个rollout的步数
            num_envs (int): 并行环境数量
            observation_space: 观察空间，包含多个传感器的观察值
            action_space: 动作空间
            recurrent_hidden_state_size (int): 循环神经网络隐藏状态大小
            use_external_memory (bool): 是否使用外部记忆机制
            external_memory_size (int): 外部记忆总大小
            external_memory_capacity (int): 外部记忆容量（每个episode）
            external_memory_dim (int): 外部记忆特征维度
            num_recurrent_layers (int): 循环神经网络层数，默认为1
        """
        # 初始化观察值存储字典，为每个传感器创建存储空间
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,  # +1是为了存储初始状态
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        # 处理SMT策略返回-1的特殊情况
        # This is introduced to handle an edge case where the
        # SMT policy returns -1 for num_recurrent_layers.
        if num_recurrent_layers < 1:
            num_recurrent_layers = 1
        
        # 初始化循环神经网络隐藏状态存储
        # 形状：(时间步数+1, 循环层数, 环境数, 隐藏状态维度)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        # 初始化奖励存储 (时间步数, 环境数, 1)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        # 初始化价值预测存储 (时间步数+1, 环境数, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        # 初始化回报值存储 (时间步数+1, 环境数, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        # 初始化动作对数概率存储 (时间步数, 环境数, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        
        # 根据动作空间类型确定动作形状
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1  # 离散动作空间
        else:
            action_shape = action_space.shape[0]  # 连续动作空间

        # 初始化动作存储 (时间步数, 环境数, 动作维度)
        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        # 初始化前一步动作存储 (时间步数+1, 环境数, 动作维度)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        
        # 如果是离散动作空间，将动作转换为长整型
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        # 初始化done mask存储，用于标记episode是否结束 (时间步数+1, 环境数, 1)
        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        # 外部记忆相关参数
        self.use_external_memory = use_external_memory
        self.em_size = external_memory_size
        self.em_capacity = external_memory_capacity
        self.em_dim = external_memory_dim
        
        # 外部记忆mask，为了向后兼容_collect_rollout_step而保留
        # This is kept outside for for backward compatibility with _collect_rollout_step
        self.em_masks = torch.zeros(num_steps + 1, num_envs, self.em_size)
        
        # 如果使用外部记忆，初始化外部记忆对象
        if use_external_memory:
            self.em = ExternalMemory(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim, num_copies=num_steps + 1
            )
        else:
            self.em = None

        # 存储rollout参数
        self.num_steps = num_steps
        self.step = 0  # 当前步数计数器

    def to(self, device):
        """将所有张量移动到指定设备（CPU或GPU）
        
        Args:
            device: 目标设备，如torch.device('cuda:0')或torch.device('cpu')
        """
        # 将所有传感器的观察值移动到指定设备
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        # 移动循环神经网络隐藏状态到指定设备
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        # 移动奖励张量到指定设备
        self.rewards = self.rewards.to(device)
        # 移动价值预测张量到指定设备
        self.value_preds = self.value_preds.to(device)
        # 移动回报值张量到指定设备
        self.returns = self.returns.to(device)
        # 移动动作对数概率张量到指定设备
        self.action_log_probs = self.action_log_probs.to(device)
        # 移动动作张量到指定设备
        self.actions = self.actions.to(device)
        # 移动前一步动作张量到指定设备
        self.prev_actions = self.prev_actions.to(device)
        # 移动done mask张量到指定设备
        self.masks = self.masks.to(device)
        # 移动外部记忆mask张量到指定设备
        self.em_masks = self.em_masks.to(device)
        # 如果使用外部记忆，也将其移动到指定设备
        if self.use_external_memory:
            self.em.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        not_done_masks,
        em_features,
    ):
        """插入一步环境交互数据到存储中
        
        Args:
            observations (dict): 当前步的观察值字典，键为传感器名称
            recurrent_hidden_states (torch.Tensor): 循环神经网络隐藏状态
            actions (torch.Tensor): 当前步执行的动作
            action_log_probs (torch.Tensor): 动作的对数概率
            value_preds (torch.Tensor): 价值函数的预测值
            rewards (torch.Tensor): 当前步获得的奖励
            not_done_masks (torch.Tensor): 标记episode是否未结束的mask
            em_features (torch.Tensor): 外部记忆特征
        """
        # 存储观察值到下一个时间步（因为需要存储初始状态）
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor].view_as(self.observations[sensor][self.step + 1])
            )
        
        # 存储循环神经网络隐藏状态到下一个时间步
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        
        # 存储当前步的动作
        self.actions[self.step].copy_(actions)
        # 将当前动作存储为下一步的前一步动作
        self.prev_actions[self.step + 1].copy_(actions)
        
        # 存储动作对数概率
        self.action_log_probs[self.step].copy_(action_log_probs)
        # 存储价值预测
        self.value_preds[self.step].copy_(value_preds)
        # 存储奖励
        self.rewards[self.step].copy_(rewards)
        # 存储done mask到下一个时间步
        self.masks[self.step + 1].copy_(not_done_masks)
        
        # 如果使用外部记忆，插入外部记忆特征
        if self.use_external_memory:
            self.em.insert(em_features, not_done_masks)
            # 更新外部记忆mask
            self.em_masks[self.step + 1].copy_(self.em.masks)

        # 步数计数器递增
        self.step = self.step + 1

    def after_update(self):
        """在模型更新后重置存储状态
        
        将最后一步的状态复制到第一步，为下一个rollout做准备。
        这确保了下一个rollout能够从当前episode的结束状态开始。
        """
        # 将最后一步的观察值复制到第一步（作为下一个rollout的初始状态）
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        # 将最后一步的循环神经网络隐藏状态复制到第一步
        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        # 将最后一步的done mask复制到第一步
        self.masks[0].copy_(self.masks[self.step])
        # 将最后一步的前一步动作复制到第一步
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        
        # 如果使用外部记忆，将最后一步的外部记忆mask复制到第一步
        if self.use_external_memory:
            self.em_masks[0].copy_(self.em_masks[self.step])
        
        # 重置步数计数器，准备下一个rollout
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        """计算回报值（returns）
        
        Args:
            next_value (torch.Tensor): 最后一步的价值预测值
            use_gae (bool): 是否使用广义优势估计（GAE）
            gamma (float): 折扣因子
            tau (float): GAE的λ参数
        """
        if use_gae:
            # 使用广义优势估计（GAE）计算回报值
            self.value_preds[self.step] = next_value
            gae = 0  # 广义优势估计值
            
            # 从后往前遍历所有步数
            for step in reversed(range(self.step)):
                # 计算时序差分误差（TD error）
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                # 更新GAE值
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                # 计算回报值 = GAE + 价值预测
                self.returns[step] = gae + self.value_preds[step]
        else:
            # 使用简单的折扣回报计算
            self.returns[self.step] = next_value
            
            # 从后往前遍历所有步数
            for step in reversed(range(self.step)):
                # 计算折扣回报：R_t = R_{t+1} * γ * mask_{t+1} + r_t
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        """生成用于循环神经网络训练的批次数据
        
        将存储的数据分批处理，用于训练循环神经网络。每个批次包含多个环境的完整序列数据。
        
        Args:
            advantages (torch.Tensor): 优势值，形状为(时间步数, 环境数, 1)
            num_mini_batch (int): 小批次数量
            
        Yields:
            tuple: 包含以下元素的元组：
                - observations_batch: 观察值批次
                - recurrent_hidden_states_batch: 循环神经网络隐藏状态批次
                - actions_batch: 动作批次
                - prev_actions_batch: 前一步动作批次
                - value_preds_batch: 价值预测批次
                - return_batch: 回报值批次
                - masks_batch: done mask批次
                - old_action_log_probs_batch: 旧动作对数概率批次
                - adv_targ: 优势目标批次
                - em_store_batch: 外部记忆存储批次（如果使用）
                - em_masks_batch: 外部记忆mask批次（如果使用）
        """
        num_processes = self.rewards.size(1)
        # 确保环境数量大于等于小批次数量
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        
        # 计算每个小批次包含的环境数量
        num_envs_per_batch = num_processes // num_mini_batch
        # 随机打乱环境顺序
        perm = torch.randperm(num_processes)
        
        # 遍历每个小批次
        for start_ind in range(0, num_processes, num_envs_per_batch):
            # 初始化批次数据列表
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            
            # 根据是否使用外部记忆初始化相应变量
            if self.use_external_memory:
                em_store_batch = []
                em_masks_batch = []
            else:
                em_store_batch = None
                em_masks_batch = None

            # 收集当前批次中每个环境的数据
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                # 收集观察值数据
                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                # 收集循环神经网络隐藏状态（只取初始状态）
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                # 收集其他数据
                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                adv_targ.append(advantages[: self.step, ind])
                
                # 如果使用外部记忆，收集外部记忆数据
                if self.use_external_memory:
                    em_store_batch.append(self.em.memory[:, : self.step, ind])
                    em_masks_batch.append(self.em_masks[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # 将所有列表堆叠成张量，形状为(T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)
            
            if self.use_external_memory:
                # 外部记忆存储张量形状为(em_size, num_steps, bs, em_dim)
                em_store_batch = torch.stack(em_store_batch, 2)
                # 外部记忆mask张量形状为(num_steps, bs, em_size)
                em_masks_batch = torch.stack(em_masks_batch, 1)

            # 循环神经网络隐藏状态形状为(num_recurrent_layers, N, -1)
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # 将(T, N, ...)形状的张量展平为(T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)
            
            if self.use_external_memory:
                em_store_batch = em_store_batch.view(-1, T * N, self.em_dim)
                em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

            # 返回当前批次的所有数据
            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                em_store_batch,
                em_masks_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        """将形状为(t, n, ...)的张量展平为(t*n, ...)
        
        这是一个静态辅助方法，用于将时间步和批次维度合并，便于批处理训练。
        
        Args:
            t (int): 张量的第一个维度（通常是时间步数）
            n (int): 张量的第二个维度（通常是批次大小）
            tensor (torch.Tensor): 需要展平的张量
            
        Returns:
            torch.Tensor: 展平后的张量，形状为(t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])

    @property
    def external_memory(self):
        """获取外部记忆存储
        
        Returns:
            torch.Tensor: 外部记忆存储张量
        """
        return self.em.memory

    @property
    def external_memory_masks(self):
        """获取外部记忆mask
        
        Returns:
            torch.Tensor: 外部记忆mask张量
        """
        return self.em_masks

    @property
    def external_memory_idx(self):
        """获取外部记忆当前索引
        
        Returns:
            int or torch.Tensor: 外部记忆当前索引
        """
        return self.em.idx


class RolloutStorageVariedExternal(RolloutStorage):
    r"""支持变化外部记忆的rollout存储类
    
    这是RolloutStorage的子类，专门用于处理外部记忆的情况。
    与基础版本不同，它维护一个em_idx向量而不是单个索引，
    这为分层策略训练提供了更大的存储灵活性。
    
    主要特点：
    1. 每个环境都有独立的外部记忆索引
    2. 支持循环缓冲区，可以处理不同长度的序列
    3. 适用于需要动态调整记忆大小的场景
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        """初始化变化外部记忆的rollout存储
        
        Args:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        if self.use_external_memory:
            num_envs = self.rewards.size(1)
            # 使用变化的外部记忆类，支持每个环境独立的索引
            self.em = ExternalMemoryVaried(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim, num_copies = self.num_steps + 1
            )
        else:
            self.em = None

    def get_em_store_and_mask(self, i, si, ei):
        """获取指定进程的起始和结束索引对应的记忆值和mask
        
        根据进程ID和起始、结束索引，获取对应的外部记忆特征和mask。
        支持循环缓冲区，可以处理索引跨越缓冲区边界的情况。
        
        Args:
            i (int): 进程ID
            si (int): 起始索引
            ei (int): 结束索引
            
        Returns:
            tuple: 包含以下元素的元组：
                - feats (torch.Tensor): 外部记忆特征，形状为(L, num_steps+1, feat_dim)
                - masks (torch.Tensor): 外部记忆mask，形状为(L,)
        """
        assert(ei != si)  # 确保起始和结束索引不同
        
        if ei > si:
            # 正常情况：起始索引小于结束索引
            feats = self.em.memory[si:ei, :, i]
            masks = self.em_masks[self.step, i, si:ei]
        else:
            # 循环情况：索引跨越缓冲区边界
            # 需要连接从起始索引到末尾和从开始到结束索引的两部分
            feats = torch.cat([self.em.memory[si:, :, i], self.em.memory[:ei, :, i]], 0)
            masks = torch.cat([self.em_masks[self.step, i, si:],
                               self.em_masks[self.step, i, :ei]], 0)
        return feats, masks


class ExternalMemoryVaried(ExternalMemory):
    """变化的外部记忆类，支持每个环境独立的索引
    
    这是ExternalMemory的子类，主要区别是每个环境都有独立的记忆索引，
    而不是所有环境共享一个索引。这使得不同环境可以有不同的记忆使用模式。
    
    主要特点：
    1. 每个环境都有独立的索引向量
    2. 支持不同长度的特征序列
    3. 更灵活的记忆管理策略
    """
    
    def __init__(self, *args, **kwargs):
        """初始化变化的外部记忆
        
        Args:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.num_envs = self.memory.size(2)
        # 为每个环境维护独立的索引
        self.idx = torch.zeros(self.num_envs).long()

    def insert(self, em_features, not_done_masks):
        """插入新的外部记忆特征（支持变化长度）
        
        Args:
            em_features (list): 每个环境的外部记忆特征列表
            not_done_masks (torch.Tensor): 标记episode是否未结束的mask
        """
        # 为每个环境更新记忆存储
        for i in range(self.num_envs):
            # 处理记忆太小无法容纳所有数据的情况
            feat_size = min(em_features[i].size(0), self.capacity)
            em_feats_i = em_features[i][-feat_size:]  # 取最后feat_size个特征
            
            # 计算起始和结束索引
            si = self.idx[i].item()
            ei = (si + feat_size) % self.total_size
            
            # 写入特征和mask
            self._write_em_store_and_mask(i, si, ei, em_feats_i)
            self.idx[i] = ei
            
        # 如果episode结束，将整个记忆mask清零
        self.masks *= not_done_masks

    def _write_em_store_and_mask(self, i, si, ei, feats):
        """为进程i写入特征和mask，处理循环包装
        
        根据进程ID和对应的起始、结束索引，写入特征和mask。
        需要处理索引环绕到缓冲区开始的情况。
        
        Args:
            i (int): 进程ID
            si (int): 起始索引，范围0到total_size-1
            ei (int): 结束索引，范围0到total_size-1
            feats (torch.Tensor): 特征张量，形状为(L, em_dim)
            
        Note:
            ei可能小于si，如果索引环绕到em_store缓冲区的开始位置
        """
        if ei == si:
            # 特殊情况：当总大小为1且索引为0时
            assert(self.total_size == 1 and ei == 0)
            ei += 1
            
        if ei > si:
            # 正常情况：起始索引小于结束索引
            self.memory[si:ei, :, i, :].copy_(feats.unsqueeze(1))
            self.masks[i, si:ei] = 1.0
        else:
            # 循环情况：索引环绕到缓冲区开始
            mi = self.total_size - si
            # 写入从起始索引到末尾的部分
            self.memory[si:, :, i, :].copy_(feats[:mi].unsqueeze(1))
            # 写入从开始到结束索引的部分
            self.memory[:ei, :, i, :].copy_(feats[mi:].unsqueeze(1))
            # 更新对应的mask
            self.masks[i, si:] = 1.0
            self.masks[i, :ei] = 1.0
            
        # 处理容量溢出：如果超过容量，移除最旧的条目
        overflow_value = int(self.masks[i].sum().item()) - self.capacity
        if overflow_value > 0:
            osi = (ei - self.capacity - overflow_value) % self.total_size
            oei = (ei - self.capacity) % self.total_size
            self._write_em_mask(i, osi, oei, 0.0)
        assert(self.masks[i].sum().item() <= self.capacity)

    def _write_em_mask(self, i, si, ei, value):
        """写入指定范围的mask值
        
        Args:
            i (int): 进程ID
            si (int): 起始索引
            ei (int): 结束索引
            value (float): 要写入的mask值
        """
        if ei > si:
            # 正常情况
            self.masks[i, si:ei] = value
        else:
            # 循环情况：需要分别处理两个范围
            self.masks[i, si:] = value
            self.masks[i, :ei] = value
