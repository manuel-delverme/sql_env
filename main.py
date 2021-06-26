import logging
import threading
import time

import experiment_buddy
import torch
import torch.multiprocessing as mp

import config
import environment  # noqa
import impala.models
from impala.core import prof
from impala.monobeast import create_buffers, act, get_batch, learn, checkpoint, create_env


if config.DEBUG:
    import multiprocessing.dummy as mp
else:
    from torch import multiprocessing as mp


def train():
    experiment_buddy.register(config_params=config.__dict__)
    writer = experiment_buddy.deploy(host="", wandb_kwargs={"mode": "disabled"})
    env = create_env(config)

    torch.manual_seed(config.seed)
    # make network
    model = impala.models.Policy(env.action_space.vocab, env.action_space.sequence_length)
    buffers = create_buffers(config.num_steps, env.observation_space.shape, len(model.output_vocab), num_buffers=config.num_buffers)
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

    learner_model = impala.models.Policy(env.action_space.vocab, env.action_space.sequence_length)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=config.learning_rate,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * config.num_steps * config.batch_size, config.total_steps) / config.total_steps

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
                step += config.num_steps * config.batch_size

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


def main():
    train()


if __name__ == "__main__":
    main()
