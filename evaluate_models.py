def test_episodes(model, env, num_episodes, max_steps = 1000):
    episode_rewards = []
    num_steps = []
    for j in range(num_episodes):
        done = False
        steps = 0
        obs = env.reset()
        episode_rewards.append(0.0)
        while(not done and steps < max_steps):
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)

            episode_rewards[-1] += reward
            steps += 1
        num_steps.append(steps)
        #print("j", j, end = " ")
    return episode_rewards, num_steps
