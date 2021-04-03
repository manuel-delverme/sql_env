import numpy as np
import torch
import torch.distributions
from stable_baselines import PPO2

import models
from env import sql_env


class Agent:
    def get_action(self, state):
        log_prob = model(state)
        action_idx = torch.argmax(q_table[f0, :])

        # probs = F.softmax(log_prob, dim=1)
        # distr = torch.distributions.Categorical(probs=probs)
        # action = distr.sample().item()
        query = models.words[action]
        return query


torch.manual_seed(1)


# model = models.SQLModel(EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)


def main():
    env = sql_env.SQLEnv(html=False)
    # agent = Agent()

    model = PPO2('MlpLstmPolicy', env, nminibatches=1, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    # Passing state=None to the predict function means
    # it is the initial state
    state = None
    # When using VecEnv, done is a vector
    done = False
    for _ in range(1000):
        # We need to pass the previous state and a mask for recurrent policies
        # to reset lstm state when a new episode begin
        action, state = model.predict(obs, state=state, mask=done)
        obs, reward, done, _ = env.step(action)
        # Note: with VecEnv, env.reset() is automatically called

        # Show the env
        env.render()

    sequence_length = 4

    num_episodes = 10000
    max_steps_per_episode = 100

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1.0
    max_exploration_rate = 1.0
    min_exploration_rate = 0.1
    exploration_decay_rate = 0.001

    action_space_size = sequence_length * len(dictionary)
    state_space_size = 100  # IDK yet
    q_table = np.zeros((state_space_size, action_space_size))

    rewards_all_episodes = []
    reference_state, _, _, _ = env.reset()
    html = False

    for episode in range(num_episodes):
        state, _, _, _ = env.reset()
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):

            exploration_rate_threshold = np.random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                agent.get_action(state)
            else:
                action_idx = np.random.randint(0, q_table.shape[1])
            action = decode_action(action_idx)

            s1, reward, done = env.step(action)
            f1 = extract_features(s1)

            q_table[f1, action] = q_table[f1, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[f1, :]))

            f0 = f1
            rewards_current_episode += reward

            if done:
                break

        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        rewards_all_episodes.append(rewards_current_episode)

    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
    count = 1000
    print("~~~~~~~Average Rewards Per Thousand Episodes~~~~~~")
    for r in rewards_per_thousand_episodes:
        print(f"{count: <5}: {np.sum(r / 1000)}")
        count += 1000

    print()
    print("~~~~~~~~~~~~~~~~~~~~~~Q-Table~~~~~~~~~~~~~~~~~~~~~~")
    print(q_table)
    return q_table, encodings


def enjoy(q_table):
    env = sql_env.SQLEnv()
    s0, _, _, _ = env.reset()
    for _ in tqdm.trange(500_000):
        f0 = extract_features(s0)
        action = np.argmax(q_table[f0, :])
        s0, _, _, _ = env.step(action)


if __name__ == "__main__":
    q_table, encodings = main()
