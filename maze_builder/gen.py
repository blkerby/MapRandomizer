from maze_builder.env import MazeBuilderEnv
from maze_builder.model import Model
from maze_builder.types import EpisodeData

import torch
import pickle
import logging
import math


# TODO: look at using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


def generate_episode_batch(env, model: Model, episode_length: int, num_candidates: int, temperature: float):
    action_list = []
    action_prob_list = []
    for j in range(episode_length):
        action_candidates = env.get_action_candidates(num_candidates)
        steps_remaining = torch.full([env.num_envs], episode_length - j,
                                     dtype=torch.float32, device=env.device)
        with torch.no_grad():
            state_value, action_value = model.forward_state_action(
                env, env.room_mask, env.room_position_x, env.room_position_y,
                action_candidates, steps_remaining)
        action_probs = torch.softmax(action_value * temperature, dim=1)
        action_index = _rand_choice(action_probs)
        selected_action_prob = action_probs[torch.arange(env.num_envs, device=env.device), action_index]
        action = action_candidates[torch.arange(env.num_envs, device=env.device), action_index, :]

        env.step(action)
        action_list.append(action)
        action_prob_list.append(selected_action_prob)

    reward_tensor = env.reward().to(torch.device('cpu'))
    action_tensor = torch.stack(action_list, dim=1).to(torch.uint8).to(torch.device('cpu'))
    action_prob_tensor = torch.stack(action_prob_list, dim=1).to(torch.device('cpu'))
    return reward_tensor, action_tensor, action_prob_tensor


def generate_episodes(base_path: str,
                      output_filename: str,
                      num_episodes: int,
                      batch_size: int,
                      num_candidates: int,
                      temperature: float,
                      save_freq: int,
                      device: torch.device):
    model = pickle.load(open(base_path + '/model.pkl', 'rb')).to(device)
    model.eval()
    env_config = model.env_config
    episode_length = len(env_config.rooms)
    env = MazeBuilderEnv(env_config.rooms, env_config.map_x, env_config.map_y, num_envs=batch_size, device=device)

    logging.info("{}".format(model))
    logging.info("Starting data generation")
    reward_list = []
    action_list = []
    num_batches = num_episodes // batch_size
    total_reward = 0
    total_reward2 = 0
    total_action_prob = 0
    cnt_episodes = 0
    for i in range(num_batches):
        env.reset()
        reward, action, action_prob = generate_episode_batch(
            env=env,
            model=model,
            episode_length=episode_length,
            num_candidates=num_candidates,
            temperature=temperature)
        reward_list.append(reward)
        action_list.append(action)

        total_reward += torch.sum(reward.to(torch.float32)).item()
        total_reward2 += torch.sum(reward.to(torch.float32) ** 2).item()
        total_action_prob += torch.sum(torch.mean(action_prob, dim=1))
        cnt_episodes += batch_size

        mean_reward = total_reward / cnt_episodes
        std_reward = math.sqrt(total_reward2 / cnt_episodes - mean_reward ** 2)
        ci_reward = std_reward * 1.96 / math.sqrt(cnt_episodes)
        mean_action_prob = total_action_prob / cnt_episodes

        logging.info("batch {}/{}: cost={:.5f} +/- {:.5f}, action_prob={:.6f}".format(
            i, num_batches, env.max_reward - mean_reward, ci_reward, mean_action_prob))

        if (i + 1) % save_freq == 0 or i == num_batches - 1:
            full_episode_data = EpisodeData(
                reward=torch.cat(reward_list, dim=0),
                action=torch.cat(action_list, dim=0),
            )
            pickle_name = base_path + output_filename
            pickle.dump(full_episode_data, open(pickle_name, 'wb'))
            logging.info("Wrote to {}".format(pickle_name))
