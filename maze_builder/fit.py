from typing import List
from maze_builder.model import Model
from maze_builder.types import FitConfig, EpisodeData

import logging
import os
import pickle
import torch


def sample_indices(num_episodes: int, episode_length: int, sample_interval: int):
    episode_index = torch.arange(num_episodes).view(-1, 1).repeat(1, episode_length).view(-1)
    step_index = torch.arange(episode_length).view(1, -1).repeat(num_episodes, 1).view(-1)
    mask = (episode_index - step_index) % sample_interval == 0
    selected_episode_index = episode_index[mask]
    selected_step_index = step_index[mask]
    return selected_episode_index, selected_step_index


def fit_model(config: FitConfig, model: Model):
    episode_data_list = []
    for filename in sorted(os.listdir(config.input_data_path)):
        if filename.startswith('data-'):
            full_path = config.input_data_path + filename
            logging.info(f"Loading {full_path}")
            episode_data = pickle.load(open(full_path, 'rb'))
            episode_data_list.append(episode_data)
    episode_data = EpisodeData(
        action=torch.cat([d.action for d in episode_data_list], dim=0),
        reward=torch.cat([d.reward for d in episode_data_list], dim=0),
    )

    eval_episode_data = EpisodeData(
        action=episode_data.action[:config.eval_num_episodes],
        reward=episode_data.reward[:config.eval_num_episodes],
    )
    eval_episode_ind, eval_step_ind = sample_indices(
        num_episodes=eval_episode_data.action.shape[0],
        episode_length=eval_episode_data.action.shape[1],
        sample_interval=config.eval_sample_interval,
    )

    train_episode_data = EpisodeData(
        action=episode_data.action[config.eval_num_episodes:(config.eval_num_episodes + config.train_num_episodes)],
        reward=episode_data.reward[config.eval_num_episodes:(config.eval_num_episodes + config.train_num_episodes)],
    )
    train_episode_ind, train_step_ind = sample_indices(
        num_episodes=train_episode_data.action.shape[0],
        episode_length=train_episode_data.action.shape[1],
        sample_interval=config.train_sample_interval,
    )
    # return episode_data