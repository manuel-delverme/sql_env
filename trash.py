episode_distances.clear()

if network_updates % config.log_query_interval == 0 and network_updates:
    data.extend([[network_updates, rollout_step, q, float(r), str(o), i["template"]] for q, r, o, i in
                 zip(queries, reward, obs, infos)])

    # for info in infos:
    #    if 'episode' in info.keys():
    #        # It's done.
    #        r = info['episode']['r']  # .detach().numpy()
    #        episode_rewards.append(r)
    #        solved = info["solved"]
    #        success_rate[info['columns']].append(solved)
    #        # agent.entropy_coef /= (1 + float(success_rate[-1]))

    #    episode_distances.append(info['similarity'])

    #  config.tb.run.log({"train_queries": wandb.Table(columns=["network_update", "rollout_step", "query", "reward", "observation", "template"], data=data)})
    config.tb.add_histogram("train/log_prob", action_logprob, global_step=network_updates)
    config.tb.add_histogram('train/log_prob_per_action',
                            np.histogram(np.arange(action_logprob.shape[0]), weights=action_logprob),
                            global_step=network_updates)
    config.tb.add_scalar("train/fps", int(total_num_steps / (end - start)), global_step=network_updates)
    config.tb.add_scalar("train/avg_rw", np.mean(episode_rewards), global_step=network_updates)
    config.tb.add_scalar("train/max_return", np.max(episode_rewards), global_step=network_updates)
    config.tb.add_scalar("train/entropy", dist_entropy, global_step=network_updates)
    config.tb.add_scalar("train/mean_distance", np.mean(episode_distances), global_step=network_updates)
    config.tb.add_scalar("train/value_loss", value_loss, global_step=network_updates)
    config.tb.add_scalar("train/action_loss", action_loss, global_step=network_updates)
    for idx, sr in enumerate(success_rate):
        if len(sr):
            config.tb.add_scalar(f"train/success_rate{idx + 1}", np.mean(sr), global_step=network_updates)

    if len(success_rate[-1]) == success_rate[-1].maxlen and np.mean(success_rate[-1]) >= 0.75:
        successes += 1
        if successes > 10:
            print("Done :)")
            return
