# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import logging
import random
from collections import namedtuple

import cogment

############ TUTORIAL STEP 4 ############
import numpy as np

##########################################
import torch
import torch.nn.functional as F
from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter, MlflowExperimentTracker
from cogment_verse_torch_agents.utils.tensors import cog_action_from_tensor, \
                                                        tensor_from_cog_continuous_action, tensor_from_cog_action, \
                                                        cog_continuous_action_from_tensor, tensor_from_cog_obs

from cogment_verse_torch_agents.utils.buffer import ReplayBuffer

from data_pb2 import (
    ActorParams,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    HumanConfig,
    HumanRole,
    MLPNetworkConfig,
    ############ TUTORIAL STEP 4 ############
    SimpleBCTrainingConfig,
    ##########################################
    SimpleBCTrainingRunConfig,
    TrialConfig,
)

SimpleBCModel = namedtuple("SimpleBCModel", ["model_id", "version_number", \
                                                    "actor", "actor_optimizer", "actor_target", \
                                                    "critic", "critic_optimizer", "critic_target"])

log = logging.getLogger(__name__)

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = torch.nn.Linear(state_dim + action_dim, 256)
        self.l5 = torch.nn.Linear(256, 256)
        self.l6 = torch.nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# pylint: disable=arguments-differ
class TD3Agent(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    @staticmethod
    async def run_async(func, *args):
        """Run a given function asynchronously in the default thread pool"""
        event_loop = asyncio.get_running_loop()
        return await event_loop.run_in_executor(None, func, *args)

    def _create(
        self,
        model_id,
        environment_specs,
        policy_network_hidden_size=64,
        **kwargs,
    ):
        policy_network=Actor(environment_specs.num_input, environment_specs.num_action)
        value_network=Critic(environment_specs.num_input, environment_specs.num_action)
        model = SimpleBCModel(
            model_id=model_id,
            version_number=1,
            actor=policy_network,
            actor_target=copy.deepcopy(policy_network),
            actor_optimizer=torch.optim.Adam(policy_network.parameters(), lr=3e-4),
            critic=value_network,
            critic_target=copy.deepcopy(value_network),
            critic_optimizer=torch.optim.Adam(value_network.parameters(), lr=3e-4),
        )

        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
        }

        return model, model_user_data

    def _load(self, model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs):
        policy_network = torch.load(model_data_f)
        assert isinstance(policy_network, torch.nn.Sequential)
        return SimpleBCModel(model_id=model_id, version_number=version_number, policy_network=policy_network)

    def _save(self, model, model_user_data, model_data_f, **kwargs):
        assert isinstance(model, SimpleBCModel)
        torch.save(model.actor, model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            model, _model_info, version_info = await self.retrieve_version(config.model_id, config.model_version)
            model_version_number = version_info["version_number"]
            log.info(f"Starting trial with model v{model_version_number}")

            # Retrieve the policy network and set it to "eval" mode
            policy_network = model.actor
            policy_network.eval()

            @torch.no_grad()
            def compute_action(event):
                with torch.no_grad():
                    #print('here'*20)
                    obs = tensor_from_cog_obs(event.observation.snapshot, dtype=self._dtype)
                    # print('obs: ', obs)
                    action = policy_network(obs.view(1, -1)).squeeze()
                    #print('action: ', action)
                    return action

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = await self.run_async(compute_action, event)
                    actor_session.do_action(cog_continuous_action_from_tensor(action))

        return {
            "simple_bc": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def sample_producer_impl(run_sample_producer_session):

            #removed for headless
            # assert run_sample_producer_session.count_actors() == 2

            total_reward = 0
            async for sample in run_sample_producer_session.get_all_samples():
                # if sample.get_trial_state() == TrialState.ENDED:
                #     break

                observation = tensor_from_cog_obs(sample.get_actor_observation(0), dtype=self._dtype)

                #print('1:', sample.get_actor_action(0))
                action=tensor_from_cog_continuous_action(sample.get_actor_action(0))

                #print('2:', action)

                reward = torch.tensor(sample.get_actor_reward(0), dtype=self._dtype)
                total_reward += reward
                done = torch.tensor(1.) if sample.get_trial_state() == TrialState.ENDED else torch.tensor(0.)


                agent_action = sample.get_actor_action(0)
                #comment out for headless
                #teacher_action = sample.get_actor_action(1)


                # print('3:', teacher_action)

                run_sample_producer_session.produce_training_sample((False, observation, action, reward, total_reward, done))

                # Check for teacher override.
                # Teacher action -1 corresponds to teacher approval,
                # i.e. the teacher considers the action taken by the agent to be correct
                # if teacher_action.continuous_action != -1:
                #    action = tensor_from_cog_continuous_action(teacher_action)
                #    run_sample_producer_session.produce_training_sample((True, observation, action, reward, done))
                # else:
                #    action = tensor_from_cog_continuous_action(agent_action)
                #    run_sample_producer_session.produce_training_sample((False, observation, action, reward, done))

        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            config = run_session.config
            # assert config.environment.specs.num_players == 1

            xp_tracker.log_params(
                config.training,
                config.environment.config,
                environment=config.environment.specs.implementation,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )

            model_id = f"{run_session.run_id}_model"

            # Initializing a model
            model, _version_info = await self.create_and_publish_initial_version(
                model_id,
                environment_specs=config.environment.specs,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )

            # Helper function to create a trial configuration
            def create_trial_config(trial_idx):
                env_params = copy.deepcopy(config.environment)
                env_params.config.seed = env_params.config.seed + trial_idx
                agent_actor_params = ActorParams(
                    name="agent_1",
                    actor_class="agent",
                    implementation="simple_bc",
                    agent_config=AgentConfig(
                        run_id=run_session.run_id,
                        model_id=model_id,
                        model_version=-1,
                        environment_specs=env_params.specs,
                    ),
                )

                teacher_actor_params = ActorParams(
                    name="web_actor",
                    actor_class="teacher_agent",
                    implementation="client",
                    human_config=HumanConfig(
                        environment_specs=env_params.specs,
                        role=HumanRole.TEACHER,
                    ),
                )

                return TrialConfig(
                    run_id=run_session.run_id,
                    environment=env_params,
                    #headless
                    # actors=[agent_actor_params],
                    #not headless
                    actors=[agent_actor_params, teacher_actor_params],
                )

            # Keep accumulated observations/actions around
            #observations = []
            #actions = []
            # dones = []
            buffer = ReplayBuffer(config.environment.specs.num_input, config.environment.specs.num_action)

            print('1: ', config.environment.specs.num_input)
            print('2: ', config.environment.specs.num_action)
            GAMMA = 0.99
            MAX_ACTION = 1.
            TAU = 0.005
            POLICY_NOISE = 0.2
            NOISE_CLIP = 0.5
            POLICY_FREQ = 2

            def train_step():

                # print('0', POLICY_FREQ)
                # print('1', TOTAL_IT)
                # global TOTAL_IT
                # TOTAL_IT += 1
                # Sample a batch of observations/actions
                #batch_indices = np.random.default_rng().integers(0, len(observations), config.training.batch_size)

                #print('observations', observations)
                #print('observations shape', observations.shape)
                # batch_obs = torch.vstack([observations[i] for i in batch_indices])
                # print('batch_obs: ', batch_obs)
                #batch_act = torch.vstack([actions[i] for i in batch_indices])
                # print('batch_act: ', batch_act)
                #batch_rew = torch.vstack([rewards[i] for i in batch_indices]).view(-1)
                # print('batch_rew: ', batch_rew)
                #batch_done = torch.vstack([dones[i] for i in batch_indices]).view(-1)
                # print('batch_done: ', batch_done)
                #batch_next_obs = torch.vstack([observations[i] for i in ((batch_indices + 1) % len(observations))])

                batch_obs, batch_act, batch_next_obs, batch_rew, batch_done = buffer.sample(config.training.batch_size)

                model.actor.train()


                # print('2')
                with torch.no_grad():
                    #print('batch_obs', batch_obs)
                    #print('batch_obs type', batch_obs.dtype)
                    #print('batch_obs shape', batch_obs.shape)
                    #print('batch_act', batch_act)
                    #print('batch_act type', batch_act.dtype)
                    #print('batch_act shape', batch_act.shape)
                    # Select action according to policy and add clipped noise

                    noise = (
                        torch.randn_like(batch_act) * POLICY_NOISE
                    ).clamp(-NOISE_CLIP, NOISE_CLIP)

                    batch_next_act = (
                        model.actor_target(batch_next_obs) + noise
                    ).clamp(-MAX_ACTION, MAX_ACTION)

                    # Compute the target Q value
                    target_Q1, target_Q2 = model.critic_target(batch_next_obs, batch_next_act)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = batch_rew + (1. - batch_done) * GAMMA * target_Q

                # print('3')
                # Get current Q estimates
                current_Q1, current_Q2 = model.critic(batch_obs, batch_act)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                model.critic_optimizer.zero_grad()
                critic_loss.backward()
                model.critic_optimizer.step()
                # print('4')

                # Delayed policy updates
                # if TOTAL_IT % POLICY_FREQ == 0:
                if random.randint(0, 1) == 0:
                # if True:

                    # Compute actor losse
                    actor_loss = -model.critic.Q1(batch_obs, model.actor(batch_obs)).mean()

                    # Optimize the actor
                    model.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    model.actor_optimizer.step()

                    # Update the frozen target models
                    for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    return actor_loss.item()

                # print('5')
                return 0

            ##########################################

            # Rollout a bunch of trials
            async for (
                ############ TUTORIAL STEP 4 ############
                step_idx,
                step_timestamp,
                ##########################################
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=[create_trial_config(trial_idx) for trial_idx in range(config.training.trial_count)],
                max_parallel_trials=config.training.max_parallel_trials,
            ):
                ############ TUTORIAL STEP 4 ############
                (_demonstration, observation, action, reward, total_reward, done) = sample
                # Can be uncommented to only use samples coming from the teacher
                # (demonstration, observation, action) = sample
                # if not demonstration:
                #     continue
                # print('observation',observation)
                # print('action',action)
                # print('reward',reward)
                # print('done',done)


                if done:
                    action = torch.tensor([0., 0.])

                    xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        total_reward=total_reward.item(),
                    )

                buffer.add(observation, action, reward, done)

                #observations.append(observation)
                #actions.append(action)
                # dones.append(done)
                if buffer.size < config.training.batch_size:
                    continue

                loss = await self.run_async(train_step)

                # Publish the newly trained version every 100 steps
                if step_idx % 100 == 0:
                    version_info = await self.publish_version(model_id, model)

                    xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        model_version_number=version_info["version_number"],
                        loss=loss,
                        total_samples=buffer.size,
                    )
                ##########################################

        return {
            "simple_bc_training": (
                sample_producer_impl,
                run_impl,
                SimpleBCTrainingRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(implementation="gym/LunarLander-v2", num_input=8, num_action=4),
                        config=EnvironmentConfig(seed=12, framestack=1, render=True, render_width=256),
                    ),
                    ############ TUTORIAL STEP 4 ############
                    training=SimpleBCTrainingConfig(
                        trial_count=100,
                        max_parallel_trials=1,
                        discount_factor=0.95,
                        learning_rate=0.01,
                    ),
                    ##########################################
                    policy_network=MLPNetworkConfig(hidden_size=64),
                ),
            )
        }
