import os
import numpy as np
from typing import Dict, Tuple
import pickle

from cogment_verse_torch_agents.third_party.hive.replays.prioritized_replay import PrioritizedReplayBuffer


class HanabiBuffer(PrioritizedReplayBuffer):
    """A Prioritized Replay buffer for the games like Hanabi with legal moves
    which need to add next_action_mask to the batch.
    """

    def __init__(
        self,
        capacity: int,
        beta: float = 0.5,
        stack_size: int = 1,
        n_step: int = 1,
        gamma: float = 0.9,
        observation_shape: Tuple = (),
        observation_dtype: type = np.uint8,
        action_shape: Tuple = (),
        action_dtype: type = np.int8,
        reward_shape: Tuple = (),
        reward_dtype: type = np.float32,
        extra_storage_types=None,
        seed: int = 42,
    ):
        super().__init__(
            capacity=capacity,
            stack_size=stack_size,
            n_step=n_step,
            gamma=gamma,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype,
            extra_storage_types=extra_storage_types,
            seed=seed,
        )

    def sample(self, batch_size):
        """Sample transitions from the buffer.
        Adding next_action_mask to the batch for environments with legal moves.
        """

        batch = super().sample(batch_size)

        batch["next_action_mask"] = self._get_from_storage(
            "action_mask",
            batch["indices"] + batch["trajectory_lengths"] - self._stack_size + 1,
            num_to_access=self._stack_size,
        )

        return batch
