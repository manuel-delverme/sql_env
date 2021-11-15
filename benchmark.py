import numpy as np
import tqdm
import ppo.model
import environment  # noqa


def main():
    env = environment.sql_env.SQLEnv()
    env.reset()
    tokens = ppo.model.Policy.output_vocab

    trials = 0
    successes = 0
    for _ in tqdm.trange(int(10e5)):
        query_ = [np.random.choice(tokens) for _ in range(env.action_space.sequence_length)]
        query = "".join(query_)
        obs, reward, done, info = env.step(query)
        if reward > 0:
            if info["solved"]:
                successes += 1
                print(trials/successes)
        trials += 1


if __name__ == "__main__":
    main()
