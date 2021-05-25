# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import threading
import time
import traceback
import typing



os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import nn
from torch.nn import functional as F

import atari_wrappers
from core import environment
from core import prof
from core import vtrace

import experiment_buddy
import config
from models import AtariNet

if config.DEBUG:
    import multiprocessing.dummy as mp
else:
    from torch import multiprocessing as mp

logging.basicConfig(format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s", level=0)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(actor_index: int, free_queue: mp.Queue, full_queue: mp.Queue, model: torch.nn.Module, buffers: Buffers,
        initial_agent_state_buffers):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(config)
        seed = actor_index ^ config.seed
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(config.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(flags, free_queue: mp.Queue, full_queue: mp.Queue, buffers: Buffers, initial_agent_state_buffers, timings,
              lock=threading.Lock()):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers}
    initial_agent_state = (torch.cat(ts, dim=1) for ts in zip(*[initial_agent_state_buffers[m] for m in indices]))
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True) for t in initial_agent_state)
    timings.time("device")
    return batch, initial_agent_state


def learn(flags, actor_model, model, batch, initial_agent_state, optimizer, scheduler, lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(learner_outputs["policy_logits"], batch["action"],
                                               vtrace_returns.pg_advantages)

        baseline_loss = flags.baseline_cost * compute_baseline_loss(vtrace_returns.vs - learner_outputs["baseline"])
        entropy_loss = flags.entropy_cost * compute_entropy_loss(learner_outputs["policy_logits"])

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train():
    experiment_buddy.register(config_params=config.__dict__)
    writer = experiment_buddy.deploy(host="", wandb_kwargs={"mode": "disabled"})
    env = create_env(config)

    torch.manual_seed(config.seed)
    model = AtariNet(env.observation_space.shape, env.action_space.n, config.use_lstm)
    buffers = create_buffers(config, env.observation_space.shape, model.num_actions)
    if config.DEBUG:
        ctx = mp
    else:
        model.share_memory()
        ctx = mp.get_context("fork")

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(config.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    free_queue = ctx.Queue()
    full_queue = ctx.Queue()

    for actor_idx in range(config.num_actors):
        actor = ctx.Process(
            target=act, args=(actor_idx, free_queue, full_queue, model, buffers, initial_agent_state_buffers),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = AtariNet(env.observation_space.shape, env.action_space.n, config.use_lstm).to(device=config.device)
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        eps=config.epsilon,
        alpha=config.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * config.unroll_length * config.batch_size, config.total_steps) / config.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, stats = 0, {}

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < config.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                config,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(config, model, learner_model, batch, agent_state, optimizer, scheduler)
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update(stats)
                for k, v in to_log.items():
                    writer.add_scalar(k, v, global_step=step)
                step += config.unroll_length * config.batch_size

        if learner_idx == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(config.num_buffers):
        free_queue.put(m)

    threads = []
    for actor_idx in range(config.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn, name="batch-and-learn-%d" % actor_idx, args=(actor_idx,))
        thread.start()
        threads.append(thread)

    try:
        while step < config.total_steps:
            start_step = step
            start_time = time.time()
            time.sleep(5)

            if step % config.save_every == 0:
                checkpoint(model, optimizer, scheduler, writer, step)

            sps = (step - start_step) / (time.time() - start_time)
            writer.add_scalar("sps", sps, global_step=step)
            mean_return = f"Return per episode: {stats.get('mean_episode_return', 0.):.1f}. "
            total_loss = stats.get("total_loss", float("inf"))
            logging.info("Steps %i @ %.1f SPS. Loss %f. %s\n", step, sps, total_loss, mean_return)

    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(config.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(model, optimizer, scheduler, writer, step)


def checkpoint(model, optimizer, scheduler, writer, step):
    if config.disable_checkpoint:
        return
    writer.add_object({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()},
        step
    )
    logging.info("Saved checkpoint to %s at %n", writer.objects_path, step)


def create_env(flags):
    import gym_minigrid
    import gym
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    return atari_wrappers.wrap_pytorch(env)


def main():
    train()


if __name__ == "__main__":
    main()
