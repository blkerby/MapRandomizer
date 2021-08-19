from typing import List
from maze_builder.env import MazeBuilderEnv
from maze_builder.model import Model
from maze_builder.types import FitConfig, EpisodeData, reconstruct_room_data
from model_average import ExponentialAverage
import shampoo

import logging
import os
import pickle
import torch
import math


def sample_indices(num_episodes: int, episode_length: int, sample_interval: int):
    episode_index = torch.arange(num_episodes).view(-1, 1).repeat(1, episode_length).view(-1)
    step_index = torch.arange(episode_length).view(1, -1).repeat(num_episodes, 1).view(-1)
    mask = (episode_index - step_index) % sample_interval == 0
    selected_episode_index = episode_index[mask]
    selected_step_index = step_index[mask]
    return selected_episode_index, selected_step_index


def extract_batch(X: torch.tensor, batch_size, batch_num) -> torch.tensor:
    start = batch_num * batch_size
    end = (batch_num + 1) * batch_size
    return X[start:end]

def forward(env: MazeBuilderEnv, model: Model, episode_data: EpisodeData,
            episode_ind: torch.tensor, step_ind: torch.tensor, device: torch.device) -> torch.tensor:
    action = episode_data.action[episode_ind, :, :]
    room_mask, room_position_x, room_position_y = reconstruct_room_data(action, step_ind, model.num_rooms)
    room_mask = room_mask.to(device)
    room_position_x = room_position_x.to(device)
    room_position_y = room_position_y.to(device)
    map = env.compute_map(room_mask, room_position_x, room_position_y)
    episode_length = episode_data.action.shape[1]
    state_value_raw_logprobs, state_value_probs, state_value_expected = model.forward_multiclass(map, room_mask, episode_length - step_ind.to(device))
    return state_value_raw_logprobs, state_value_probs, state_value_expected

def eval(fit_config: FitConfig, model: Model, env: MazeBuilderEnv, eval_episode_data: EpisodeData,
         eval_episode_ind: torch.tensor, eval_step_ind: torch.tensor):
    device = next(iter(model.parameters())).device
    num_eval_batches = (eval_episode_ind.shape[0] + fit_config.eval_batch_size - 1) // fit_config.eval_batch_size

    eval_loss = torch.zeros([len(fit_config.eval_loss_objs), eval_episode_data.action.shape[0]], dtype=torch.float32)
    eval_cnt = torch.zeros([eval_episode_data.action.shape[0]], dtype=torch.float32)
    ones = torch.ones([fit_config.eval_batch_size], dtype=torch.float32)
    for i in range(num_eval_batches):
        batch_episode_ind = extract_batch(eval_episode_ind, fit_config.eval_batch_size, i)
        batch_step_ind = extract_batch(eval_step_ind, fit_config.eval_batch_size, i)
        with torch.no_grad():
            batch_state_value_raw_logprobs, batch_state_value_probs, batch_state_value_expected = forward(
                env, model, eval_episode_data, batch_episode_ind, batch_step_ind, device)
        batch_reward = eval_episode_data.reward[batch_episode_ind].to(device)
        for j, loss_obj in enumerate(fit_config.eval_loss_objs):
            loss_obj.reduction = 'none'
            if isinstance(loss_obj, torch.nn.CrossEntropyLoss):
                pred = batch_state_value_raw_logprobs
                target = batch_reward
            else:
                pred = batch_state_value_expected
                target = batch_reward.to(torch.float32)
            loss = loss_obj(pred, target)
            eval_loss[j, :].scatter_add_(dim=0, index=batch_episode_ind, src=loss.to('cpu'))
        eval_cnt.scatter_add_(dim=0, index=batch_episode_ind, src=ones)
    eval_loss /= eval_cnt
    eval_loss_mean = torch.mean(eval_loss, dim=1)
    eval_loss_std = torch.std(eval_loss, dim=1)
    eval_loss_ci = eval_loss_std / math.sqrt(eval_loss.shape[1]) * 1.96
    eval_fmt = ', '.join('{:#.5g} +/- {:#.2g}'.format(eval_loss_mean[i], eval_loss_ci[i])
                         for i in range(eval_loss_mean.shape[0]))
    return eval_fmt


def fit_model(fit_config: FitConfig, model: Model, baseline_model: Model):
    episode_data_list = []
    for filename in sorted(os.listdir(fit_config.input_data_path)):
        if filename.startswith('data-'):
            full_path = fit_config.input_data_path + filename
            logging.info(f"Loading {full_path}")
            episode_data = pickle.load(open(full_path, 'rb'))
            episode_data_list.append(episode_data)
    logging.info("Concatenating data")
    episode_data = EpisodeData(
        action=torch.cat([d.action for d in episode_data_list], dim=0),
        reward=torch.cat([d.reward for d in episode_data_list], dim=0),
    )
    del episode_data_list

    total_episodes_to_use = fit_config.eval_num_episodes + fit_config.train_num_episodes
    logging.info("Loaded {} episodes, to use {} ({} for eval, {} for train)".format(
        episode_data.reward.shape[0], total_episodes_to_use, fit_config.eval_num_episodes, fit_config.train_num_episodes
    ))
    assert episode_data.reward.shape[0] >= total_episodes_to_use
    logging.info("Shuffling data")
    eval_episode_data = EpisodeData(
        action=episode_data.action[:fit_config.eval_num_episodes],
        reward=episode_data.reward[:fit_config.eval_num_episodes],
    )
    eval_episode_ind, eval_step_ind = sample_indices(
        num_episodes=eval_episode_data.action.shape[0],
        episode_length=eval_episode_data.action.shape[1],
        sample_interval=fit_config.eval_sample_interval,
    )

    train_episode_data = EpisodeData(
        action=episode_data.action[fit_config.eval_num_episodes:(fit_config.eval_num_episodes + fit_config.train_num_episodes)],
        reward=episode_data.reward[fit_config.eval_num_episodes:(fit_config.eval_num_episodes + fit_config.train_num_episodes)],
    )
    train_episode_ind, train_step_ind = sample_indices(
        num_episodes=train_episode_data.action.shape[0],
        episode_length=train_episode_data.action.shape[1],
        sample_interval=fit_config.train_sample_interval,
    )
    torch.manual_seed(fit_config.train_shuffle_seed)
    train_perm = torch.randperm(train_episode_ind.shape[0])
    train_episode_ind = train_episode_ind[train_perm]
    train_step_ind = train_step_ind[train_perm]

    device = next(iter(model.parameters())).device
    env = MazeBuilderEnv(rooms=model.env_config.rooms,
                         map_x=model.env_config.map_x,
                         map_y=model.env_config.map_y,
                         num_envs=0,
                         device=device)
    num_train_batches = train_episode_ind.shape[0] // fit_config.train_batch_size

    grad_scaler = torch.cuda.amp.GradScaler()
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                                 lr=fit_config.optimizer_learning_rate0,
    #                                 alpha=fit_config.optimizer_alpha)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=fit_config.optimizer_learning_rate0,
                                 betas=(fit_config.optimizer_beta, fit_config.optimizer_alpha))
    # optimizer = shampoo.Shampoo(model.parameters(),
    #                             lr=fit_config.optimizer_learning_rate0,
    #                             update_freq=10,
    #                             epsilon=1.0,
    #                             beta=0.99)
    average_parameters = ExponentialAverage(model.all_param_data(), beta=fit_config.polyak_ema_beta)
    i = 0

    total_loss = 0.0
    total_loss_cnt = 0
    logging.info(fit_config)
    logging.info(model)
    logging.info(optimizer)
    eval_fmt = eval(fit_config, baseline_model, env, eval_episode_data, eval_episode_ind, eval_step_ind)
    logging.info("Baseline test={}".format(eval_fmt))
    logging.info("Training")
    while i < num_train_batches:
        frac = i / num_train_batches
        lr = fit_config.optimizer_learning_rate0 * (fit_config.optimizer_learning_rate1 / fit_config.optimizer_learning_rate0) ** frac
        optimizer.param_groups[0]['lr'] = lr

        if fit_config.sam_scale is not None:
            saved_params = [param.data.clone() for param in model.parameters()]
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * fit_config.sam_scale
            # model.project()

        batch_episode_ind = extract_batch(train_episode_ind, fit_config.train_batch_size, i)
        batch_step_ind = extract_batch(train_step_ind, fit_config.train_batch_size, i)
        batch_state_value_raw_logprobs, batch_state_value_probs, batch_state_value_expected = forward(
            env, model, train_episode_data, batch_episode_ind, batch_step_ind, device)
        batch_reward = train_episode_data.reward[batch_episode_ind].to(device)
        if isinstance(fit_config.train_loss_obj, torch.nn.CrossEntropyLoss):
            pred = batch_state_value_raw_logprobs
            target = batch_reward
        else:
            pred = batch_state_value_expected
            target = batch_reward.to(torch.float32)
        loss = fit_config.train_loss_obj(pred, target)

        optimizer.zero_grad()

        # with torch.autograd.detect_anomaly():
        grad_scaler.scale(loss).backward()

        if fit_config.sam_scale is not None:
            for j, param in enumerate(model.parameters()):
                param.data.copy_(saved_params[j])

        grad_scaler.step(optimizer)
        grad_scaler.update()
        model.decay(fit_config.weight_decay * lr)
        model.project()
        average_parameters.update(model.all_param_data())
        total_loss += loss.item()
        total_loss_cnt += 1

        i += 1
        if i % fit_config.eval_freq == 0:
            model.eval()
            with average_parameters.average_parameters(model.all_param_data()):
                eval_fmt = eval(fit_config, model, env, eval_episode_data, eval_episode_ind, eval_step_ind)
            logging.info("{}/{}: train={:#.5g}, test={}, lr={:.6f}".format(
                i, num_train_batches, total_loss / total_loss_cnt, eval_fmt, lr))
            total_loss_cnt = 0
            total_loss = 0.0
            model.train()

    average_parameters.use_averages(model.all_param_data())
    eval_fmt = eval(fit_config, model, env, eval_episode_data, eval_episode_ind, eval_step_ind)
    logging.info("{}/{}: final test={}".format(i, num_train_batches, eval_fmt))

    # return episode_data