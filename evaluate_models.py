import constants
def test_episodes(model, env, num_episodes):
    episode_rewards = []
    num_steps = []
    successes = []
    successes2 = []
    for j in range(num_episodes):
        done = False
        steps = 0
        obs = env.reset()
        episode_rewards.append(0.0)
        while(not done and steps < constants.max_episode_steps):
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)

            episode_rewards[-1] += reward
            steps += 1
        successes.append(done)
        successes2.append(reward == 1)
        num_steps.append(steps)
    return episode_rewards, num_steps, successes, successes2
