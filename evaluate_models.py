def test_episodes(model, env, num_episodes):
    episode_rewards = []
    num_steps = []
    successes = []
    for j in range(num_episodes):
        done = False
        steps = 0
        obs = env.reset()
        episode_rewards.append(0.0)
        while(not done):
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)

            episode_rewards[-1] += reward
            steps += 1
        successes.append(reward == 1)
        num_steps.append(steps)
    return episode_rewards, num_steps, successes2


def look_at_an_episode(model, env):
    done = False
    steps = 0
    obs = env.reset()
    final_reward = 0
    import structured.generate_actions
    actions = structured.generate_actions.generate_actions()
    while(not done):
        action, _states = model.predict(obs, deterministic = True)
        print(action, actions[action], end = ":")
        _obs, reward, done, resp = env.step(action)
        print(reward, resp["msg"])
        final_reward += reward

        steps += 1

    print("steps", steps, "reward", final_reward)
