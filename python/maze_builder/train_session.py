from typing import Optional, List
import copy
import torch
import torch.nn.functional as F
from maze_builder.model import RoomTransformerModel
from maze_builder.env import MazeBuilderEnv, compute_cycle_costs
from maze_builder.replay import ReplayBuffer
from maze_builder.types import EpisodeData, TrainingData
from model_average import ExponentialAverage
import concurrent.futures
import dataclasses


# TODO: try using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


# def compute_mse_loss(log_probs, labels):
#     probs = torch.where(log_probs >= 0, log_probs + 1,
#                         torch.exp(torch.clamp_max(log_probs, 0.0)))  # Clamp is "no-op" but avoids non-finite gradients
#     return (probs - labels) ** 2

# action_indexes = []

@dataclasses.dataclass
class Predictions:
    door_connects: torch.tensor
    missing_connects: torch.tensor
    door_balance: torch.tensor
    save_dist: torch.tensor
    graph_diam: torch.tensor
    mc_dist: torch.tensor
    toilet_good: torch.tensor


class TrainingSession():
    def __init__(self, envs: List[MazeBuilderEnv],
                 action_model: RoomTransformerModel,
                 balance_model: torch.nn.Module,
                 action_optimizer: torch.optim.Optimizer,
                 balance_optimizer: torch.optim.Optimizer,
                 data_path: str,
                 ema_beta: float,
                 episodes_per_file: int,
                 decay_amount: float
                 ):
        self.envs = envs
        self.action_model = action_model
        self.balance_model = balance_model
        self.action_optimizer = action_optimizer
        self.balance_optimizer = balance_optimizer
        self.action_average_parameters = ExponentialAverage(action_model.all_param_data(), beta=ema_beta)
        self.balance_average_parameters = ExponentialAverage(balance_model.parameters(), beta=ema_beta)
        self.num_rounds = 0
        self.decay_amount = decay_amount
        self.action_grad_scaler = torch.cuda.amp.GradScaler()
        self.replay_buffer = ReplayBuffer(len(self.envs[0].rooms),
                                          storage_device=torch.device('cpu'),
                                          data_path=data_path,
                                          episodes_per_file=episodes_per_file)

        self.total_step_remaining_gen = 0.0
        self.total_step_remaining_train = 0.0
        self.verbose = False

    def compute_door_stats_entropy(self, counts):
        if counts is None:
            return float('nan')
        total_doors = 0
        total_entropy = 0.0
        for cnt in counts:
            total_doors += cnt.shape[0]
            p = cnt / torch.sum(cnt, dim=1, keepdim=True)
            total_entropy += torch.sum(p * -torch.log(p + 1e-15)).item()

            cnt = torch.transpose(cnt, 0, 1)
            total_doors += cnt.shape[0]
            p = cnt / torch.sum(cnt, dim=1, keepdim=True)
            total_entropy += torch.sum(p * -torch.log(p + 1e-15)).item()
        return total_entropy / total_doors

    def center_matrix(self, A, wt):
        wt_mean_row = torch.sum(A * wt, dim=0, keepdim=True) / (torch.sum(wt, dim=0, keepdim=True) + 1e-15)
        A = A - wt_mean_row

        wt_mean_col = torch.sum(A * wt, dim=1, keepdim=True) / (torch.sum(wt, dim=1, keepdim=True) + 1e-15)
        A = A - wt_mean_col

        return A

    def get_door_connect_entropy(self, num_examples):
        stats_left_list = []
        stats_down_list = []
        for env in self.envs:
            batch_stats_left, batch_stats_down = env.get_door_connect_stats(
                env.room_mask[:num_examples], env.room_position_x[:num_examples], env.room_position_y[:num_examples])
            stats_left_list.append(batch_stats_left.to(torch.float).to(self.envs[0].device))
            stats_down_list.append(batch_stats_down.to(torch.float).to(self.envs[0].device))

        stats_left = torch.sum(torch.stack(stats_left_list, dim=0), dim=0)
        stats_down = torch.sum(torch.stack(stats_down_list, dim=0), dim=0)
        ent = self.compute_door_stats_entropy((stats_left, stats_down))
        return ent

    def compute_reward(self, door_connects, missing_connects, use_connectivity):
        reward = torch.sum(~door_connects, dim=1) // 2
        if use_connectivity:
            reward += torch.sum(~missing_connects, dim=1)
        return reward

    def get_output_room_ids(self):
        # Order must match get_preds
        room_ids = []
        env = self.envs[0]

        # door connects
        room_dir_list = [env.room_left, env.room_right, env.room_down, env.room_up]
        for room_dir in room_dir_list:
            room_ids.extend(room_dir[:, 0].tolist())

        # missing connects
        room_ids.extend(env.part_room_id[env.missing_connection_src].tolist())

        # toilet good (use global token)
        room_ids.append(len(env.rooms))

        # door balance
        for room_dir in room_dir_list:
            room_ids.extend(room_dir[:, 0].tolist())

        # save balance
        room_ids.extend(env.part_room_id[env.non_potential_save_idxs])

        # graph diameter (use global token)
        room_ids.append(len(env.rooms))

        # missing connect return distance
        room_ids.extend(env.part_room_id[env.missing_connection_src].tolist())

        return room_ids

    def get_preds(self, raw_preds):
        # Order must match get_output_room_ids
        env = self.envs[0]
        output_sizes = [
            env.num_doors,
            env.num_missing_connects,
            1,  # toilet_good
            env.num_doors,  # door balance
            env.non_potential_save_idxs.shape[0],
            1,  # graph diam
            env.num_missing_connects,
        ]
        assert raw_preds.shape[1] == sum(output_sizes)
        preds = []
        col = 0
        for size in output_sizes:
            preds.append(raw_preds[:, col:(col + size)])
            col += size
        return Predictions(
            door_connects=preds[0],
            missing_connects=preds[1],
            toilet_good=preds[2][:, 0],
            door_balance=preds[3],
            save_dist=preds[4],
            graph_diam=preds[5][:, 0],
            mc_dist=preds[6],
        )

    def forward_action(self, model, room_mask, room_position_x, room_position_y, map_door_ids, action_candidates,
                       steps_remaining, temperature,
                       env_id, balance_coef: float, save_dist_coef: float, graph_diam_coef: float,
                       mc_dist_coef: torch.tensor,
                       toilet_good_coef: float,
                       executor):
        # print({k: v.shape for k, v in locals().items() if hasattr(v, 'shape')})
        #
        # torch.cuda.synchronize()
        # logging.info("Processing candidate data")
        num_envs = room_mask.shape[0]
        num_candidates = action_candidates.shape[1]
        num_rooms = len(self.envs[0].rooms)
        action_room_id = action_candidates[:, :, 0]
        action_x = action_candidates[:, :, 1]
        action_y = action_candidates[:, :, 2]
        action_room_door_id = action_candidates[:, :, 4]
        valid = (action_room_id != len(self.envs[0].rooms) - 1)
        round_frac = torch.ones([num_envs], device=action_candidates.device,
                                dtype=torch.float32)

        all_mc_dist_coef = mc_dist_coef.unsqueeze(1).repeat(1, num_candidates)

        room_door_id_flat = action_room_door_id.view(num_envs * num_candidates)
        mc_dist_coef_flat = all_mc_dist_coef.view(num_envs * num_candidates)
        valid_flat = valid.view(num_envs * num_candidates)
        valid_flat_ind = torch.nonzero(valid_flat)[:, 0]

        action_room_id_flat = action_room_id.view(num_envs * num_candidates)
        action_x_flat = action_x.view(num_envs * num_candidates)
        action_y_flat = action_y.view(num_envs * num_candidates)

        action_env_id_valid = valid_flat_ind // num_candidates
        action_room_id_valid = action_room_id_flat[valid_flat_ind]
        action_x_valid = action_x_flat[valid_flat_ind]
        action_y_valid = action_y_flat[valid_flat_ind]
        room_door_id_valid = room_door_id_flat[valid_flat_ind]

        mc_dist_coef_valid = mc_dist_coef_flat[valid_flat]

        # mc_dist_coef_valid = mc_dist_coef[action_env_id_valid]

        env = self.envs[env_id]
        # map_flat = env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)
        # map_valid = env.compute_map(room_mask_valid, room_position_x_valid, room_position_y_valid)

        # torch.cuda.synchronize()
        # logging.info("Model forward")
        # flat_raw_logodds, _, flat_expected = model.forward_multiclass(
        #     map_flat, room_mask_flat, room_position_x_flat, room_position_y_flat, steps_remaining_flat, round_frac_flat,
        #     temperature_flat, env)

        raw_preds_valid = model.forward_multiclass(
            room_mask, room_position_x, room_position_y,
            map_door_ids, action_env_id_valid, room_door_id_valid,
            steps_remaining, round_frac,
            temperature, mc_dist_coef, self.envs[env_id], compute_state_value=False)

        preds = self.get_preds(raw_preds_valid)

        logodds_valid = torch.cat([preds.door_connects, preds.missing_connects], dim=1)
        logprobs_valid = -torch.logaddexp(-logodds_valid, torch.zeros_like(logodds_valid))
        expected_valid = torch.sum(logprobs_valid, dim=1)  # / 2

        pred_toilet_good_logprobs = -torch.logaddexp(-preds.toilet_good, torch.zeros_like(preds.toilet_good))

        # print("score idx: ", num_logodds + num_save_dist, "num_logodds =", num_logodds, "num_save_dist =", num_save_dist)

        # Note: for steps with no valid candidates (i.e. when no more rooms can be placed), we won't generate any
        # predictions, and the test_loss will be computed just based on these zero log-odds filler values.
        logodds_flat = torch.zeros([num_envs * num_candidates, logodds_valid.shape[-1]], device=logodds_valid.device)
        logodds_flat[valid_flat_ind, :] = logodds_valid

        # Adjust score using additional term to encourage balanced distribution of potential save station rooms
        # expected_valid = expected_valid - pred_save_dist * save_dist_coef

        # Adjust the scores to disfavor door connections which occurred frequently in the past:
        # door_connect_cost1 = self.door_connect_adjust.to(expected_valid.device)[map_door_id_valid, room_door_id_valid]
        # door_connect_cost2 = self.door_connect_adjust.to(expected_valid.device)[room_door_id_valid, map_door_id_valid]

        # door_connect_cost = self.compute_candidate_penalties(
        #     room_mask, room_position_x, room_position_y, action_env_id_valid, action_room_id_valid, action_x_valid, action_y_valid, env_id,
        #     adjust_left_right, adjust_down_up)
        door_connect_cost = 0.0  # TODO: compute this using model predictions
        balance_cost = torch.sum(torch.sigmoid(preds.door_balance), dim=1)
        save_dist_cost = torch.sum(preds.save_dist, dim=1)
        mc_dist_cost = torch.sum(preds.mc_dist, dim=1)
        expected_valid = expected_valid - door_connect_cost - balance_cost * balance_coef - save_dist_cost * save_dist_coef - preds.graph_diam * graph_diam_coef - mc_dist_cost * mc_dist_coef_valid + pred_toilet_good_logprobs * toilet_good_coef

        expected_flat = torch.full([num_envs * num_candidates], -1e15, device=logodds_valid.device)
        expected_flat[valid_flat_ind] = expected_valid

        raw_logodds = logodds_flat.view(num_envs, num_candidates, -1)
        expected = expected_flat.view(num_envs, num_candidates)
        return expected, raw_logodds

    def generate_round_inner(self, model, episode_length: int, num_candidates_min: float, num_candidates_max: float,
                             temperature: torch.tensor,
                             temperature_decay: float, explore_eps: torch.tensor,
                             env_id, balance_coef: float, save_dist_coef: float, graph_diam_coef: float,
                             mc_dist_coef: torch.tensor,
                             toilet_good_coef: float,
                             render, executor) -> EpisodeData:
        with (torch.no_grad()):
            device = self.envs[env_id].device
            env = self.envs[env_id]
            env.reset()
            selected_raw_logodds_list = []
            action_list = []
            prob_list = []
            prob0_list = []
            cand_count_list = []
            map_door_id_list = []
            room_door_id_list = []
            model.eval()
            temperature = temperature.to(device)
            mc_dist_coef = mc_dist_coef.to(device)
            # explore_eps = explore_eps.to(device).unsqueeze(1)
            # torch.cuda.synchronize()
            # logging.debug("Averaging parameters")

            for j in range(episode_length):
                if render:
                    env.render()

                frac = j / (episode_length - 1)
                num_candidates = int(round(num_candidates_min * (num_candidates_max / num_candidates_min) ** frac))
                num_candidates = min(num_candidates,
                                     episode_length - j)  # The number of candidates won't be more than the number of steps remaining.
                if j == 0:
                    # Place the first room (Landing Site) uniformly randomly:
                    num_candidates = 1

                # torch.cuda.synchronize()
                # logging.debug("Getting candidates")
                action_candidates, map_door_ids = env.get_action_candidates(
                    num_candidates, env.room_mask, env.room_position_x,
                    env.room_position_y, verbose=self.verbose)
                # action_candidates = env.get_all_action_candidates(env.room_mask, env.room_position_x, env.room_position_y)
                steps_remaining = torch.full([env.num_envs], episode_length - j,
                                             dtype=torch.float32, device=device)

                if num_candidates == 1:
                    # action_expected = torch.zeros([env.num_envs, num_candidates], dtype=torch.float32, device=device)
                    raw_logodds = torch.zeros([env.num_envs, num_candidates, env.num_doors + env.num_missing_connects],
                                              dtype=torch.float32, device=device)
                    probs = torch.ones([env.num_envs, 1], dtype=torch.float32, device=device)
                    action_index = torch.zeros([env.num_envs], dtype=torch.long, device=device)
                else:
                    # print("inner", env_id, j, env.device, model.state_value_lin.weight.device)
                    action_expected, raw_logodds = self.forward_action(
                        model, env.room_mask, env.room_position_x, env.room_position_y, map_door_ids,
                        action_candidates, steps_remaining, temperature, env_id, balance_coef, save_dist_coef,
                        graph_diam_coef,
                        mc_dist_coef, toilet_good_coef, executor)
                    curr_temperature = temperature * temperature_decay ** (j / (episode_length - 1))
                    probs = torch.softmax(action_expected / torch.unsqueeze(curr_temperature, 1), dim=1)
                    action_index = _rand_choice(probs)

                # action_expected = torch.where(action_candidates[:, :, 0] == len(env.rooms) - 1,
                #                               torch.full_like(action_expected, -1e15),
                #                               action_expected)  # Give dummy move negligible probability except where it is the only choice
                #
                # print(action_expected)

                candidate_count = torch.sum(probs > 0, dim=1)
                # candidate_count1 = torch.sum(action_candidates[:, :, 0] != len(env.rooms) - 1, dim=1)
                # candidate_count = torch.clamp_min(torch.sum(action_candidates[:, :, 0] != len(env.rooms) - 1, dim=1), 1)
                # explore_probs = torch.where(action_candidates[:, :, 0] != len(env.rooms) - 1,
                #                             1 / candidate_count1.unsqueeze(1),
                #                             torch.zeros_like(probs))
                # new_probs = explore_eps * explore_probs + (1 - explore_eps) * probs
                # probs = torch.where(candidate_count1.unsqueeze(1) > 0, new_probs, probs)
                # action_indexes.append(action_index)  # TODO: remove this
                selected_prob = probs[torch.arange(env.num_envs, device=device), action_index]
                selected_prob0 = selected_prob * candidate_count
                action = action_candidates[torch.arange(env.num_envs, device=device), action_index, :3]
                room_door_ids = action_candidates[torch.arange(env.num_envs, device=device), action_index, 4]
                selected_raw_logodds = raw_logodds[torch.arange(env.num_envs, device=device), action_index, :]

                env.step(action)
                action_list.append(action.to('cpu'))
                selected_raw_logodds_list.append(selected_raw_logodds.to('cpu'))
                prob_list.append(selected_prob.to('cpu'))
                prob0_list.append(selected_prob0.to('cpu'))
                cand_count_list.append(candidate_count.to(torch.float32).to('cpu'))
                map_door_id_list.append(map_door_ids.to(torch.int16).to('cpu'))
                room_door_id_list.append(room_door_ids.to(torch.int16).to('cpu'))

            # torch.cuda.synchronize(device)
            door_connects_tensor = env.current_door_connects().to('cpu')
            part_adjacency_matrix = env.compute_part_adjacency_matrix(env.room_mask, env.room_position_x,
                                                                      env.room_position_y)
            missing_connects_tensor = env.compute_missing_connections(part_adjacency_matrix).to('cpu')
            distance_matrix = env.compute_distance_matrix(part_adjacency_matrix)
            save_distances = env.compute_save_distances(distance_matrix).to('cpu')
            graph_diameter = env.compute_graph_diameter(distance_matrix).to('cpu')
            mc_distances = env.compute_mc_distances(distance_matrix).to('cpu')
            toilet_good = env.compute_toilet_good(env.room_mask, env.room_position_x, env.room_position_y).to('cpu')
            reward_tensor = self.compute_reward(door_connects_tensor, missing_connects_tensor, use_connectivity=True)
            selected_raw_logodds_tensor = torch.stack(selected_raw_logodds_list, dim=1)
            action_tensor = torch.stack(action_list, dim=1)
            prob_tensor = torch.mean(torch.stack(prob_list, dim=1), dim=1)
            prob0_tensor = torch.mean(torch.stack(prob0_list, dim=1), dim=1)
            cand_count_tensor = torch.mean(torch.stack(cand_count_list, dim=1), dim=1)
            map_door_ids_tensor = torch.stack(map_door_id_list, dim=1)
            room_door_ids_tensor = torch.stack(room_door_id_list, dim=1)

            selected_raw_logodds_flat = selected_raw_logodds_tensor.view(env.num_envs * episode_length,
                                                                         selected_raw_logodds_tensor.shape[-1])
            # reward_flat = reward_tensor.view(env.num_envs, 1).repeat(1, episode_length).view(-1)
            door_connects_flat = door_connects_tensor.view(env.num_envs, 1, -1).repeat(1, episode_length, 1).view(
                env.num_envs * episode_length, -1)
            missing_connects_flat = missing_connects_tensor.view(env.num_envs, 1, -1).repeat(1, episode_length, 1).view(
                env.num_envs * episode_length, -1)
            all_outputs_flat = torch.cat([door_connects_flat, missing_connects_flat], dim=1)

            loss_flat = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(selected_raw_logodds_flat,
                                                                                        all_outputs_flat.to(
                                                                                            selected_raw_logodds_flat.dtype),
                                                                                        reduction='none'), dim=1)
            # loss_flat = torch.mean(compute_mse_loss(state_raw_logodds_flat, all_outputs_flat.to(state_raw_logodds_flat.dtype)), dim=1)
            loss = loss_flat.view(env.num_envs, episode_length)
            episode_loss = torch.mean(loss, dim=1)

            episode_data = EpisodeData(
                reward=reward_tensor,
                door_connects=door_connects_tensor,
                missing_connects=missing_connects_tensor,
                save_distances=save_distances,
                graph_diameter=graph_diameter,
                mc_distances=mc_distances,
                toilet_good=toilet_good,
                cycle_cost=None,  # populated later in generate_round_model
                action=action_tensor.to(torch.uint8),
                map_door_id=map_door_ids_tensor,
                room_door_id=room_door_ids_tensor,
                prob=prob_tensor,
                prob0=prob0_tensor,
                cand_count=cand_count_tensor,
                temperature=temperature.to('cpu'),
                mc_dist_coef=mc_dist_coef.to('cpu'),
                test_loss=episode_loss,
            )
            return episode_data

    def generate_round_model(self, model, episode_length: int, num_candidates_min: float, num_candidates_max: float,
                             temperature: torch.tensor,
                             temperature_decay: float,
                             explore_eps: torch.tensor,
                             compute_cycles: bool,
                             balance_coef: float,
                             save_dist_coef: float,
                             graph_diam_coef: float,
                             mc_dist_coef: torch.tensor,
                             toilet_good_coef: float,
                             executor: concurrent.futures.ThreadPoolExecutor,
                             cpu_executor: concurrent.futures.ProcessPoolExecutor,
                             render=False) -> EpisodeData:
        futures_list = []
        model_list = [copy.deepcopy(model).to(env.device) for env in self.envs]
        for i, env in enumerate(self.envs):
            model = model_list[i]
            # print("gen", i, env.device, model.state_value_lin.weight.device)
            future = executor.submit(lambda i=i, model=model: self.generate_round_inner(
                model, episode_length, num_candidates_min, num_candidates_max, temperature, temperature_decay,
                explore_eps, render=render,
                env_id=i, balance_coef=balance_coef, save_dist_coef=save_dist_coef, graph_diam_coef=graph_diam_coef,
                mc_dist_coef=mc_dist_coef,
                toilet_good_coef=toilet_good_coef, executor=executor))
            futures_list.append(future)

        episode_data_list = []
        for env, future in zip(self.envs, futures_list):
            episode_data = future.result()
            room_adjacency_matrix = env.compute_room_adjacency_matrix(env.room_mask, env.room_position_x,
                                                                      env.room_position_y)
            if compute_cycles:
                cycle_cost_tensor = compute_cycle_costs(room_adjacency_matrix, cpu_executor)
            else:
                cycle_cost_tensor = torch.full([env.num_envs], float('nan'))
            episode_data.cycle_cost = cycle_cost_tensor
            episode_data_list.append(episode_data)
        # for env in self.envs:
        #     if env.room_mask.is_cuda:
        #         torch.cuda.synchronize(env.device)
        return EpisodeData(
            reward=torch.cat([d.reward for d in episode_data_list], dim=0),
            door_connects=torch.cat([d.door_connects for d in episode_data_list], dim=0),
            missing_connects=torch.cat([d.missing_connects for d in episode_data_list], dim=0),
            save_distances=torch.cat([d.save_distances for d in episode_data_list], dim=0),
            graph_diameter=torch.cat([d.graph_diameter for d in episode_data_list], dim=0),
            mc_distances=torch.cat([d.mc_distances for d in episode_data_list], dim=0),
            toilet_good=torch.cat([d.toilet_good for d in episode_data_list], dim=0),
            cycle_cost=torch.cat([d.cycle_cost for d in episode_data_list], dim=0),
            action=torch.cat([d.action for d in episode_data_list], dim=0),
            map_door_id=torch.cat([d.map_door_id for d in episode_data_list], dim=0),
            room_door_id=torch.cat([d.room_door_id for d in episode_data_list], dim=0),
            prob=torch.cat([d.prob for d in episode_data_list], dim=0),
            prob0=torch.cat([d.prob0 for d in episode_data_list], dim=0),
            cand_count=torch.cat([d.cand_count for d in episode_data_list], dim=0),
            temperature=torch.cat([d.temperature for d in episode_data_list], dim=0),
            mc_dist_coef=torch.cat([d.mc_dist_coef for d in episode_data_list], dim=0),
            test_loss=torch.cat([d.test_loss for d in episode_data_list], dim=0),
        )

    def generate_round(self, episode_length: int, num_candidates_min: float, num_candidates_max: float,
                       temperature: torch.tensor,
                       temperature_decay: float,
                       explore_eps: torch.tensor,
                       compute_cycles: bool,
                       balance_coef: float,
                       save_dist_coef: float,
                       graph_diam_coef: float,
                       mc_dist_coef: torch.tensor,
                       toilet_good_coef: float,
                       executor: Optional[concurrent.futures.ThreadPoolExecutor],
                       cpu_executor: concurrent.futures.ProcessPoolExecutor,
                       render=False) -> EpisodeData:
        with self.action_average_parameters.average_parameters(self.action_model.all_param_data()):
            return self.generate_round_model(model=self.action_model,
                                             episode_length=episode_length,
                                             num_candidates_min=num_candidates_min,
                                             num_candidates_max=num_candidates_max,
                                             temperature=temperature,
                                             temperature_decay=temperature_decay,
                                             explore_eps=explore_eps,
                                             compute_cycles=compute_cycles,
                                             balance_coef=balance_coef,
                                             save_dist_coef=save_dist_coef,
                                             graph_diam_coef=graph_diam_coef,
                                             mc_dist_coef=mc_dist_coef,
                                             toilet_good_coef=toilet_good_coef,
                                             executor=executor,
                                             cpu_executor=cpu_executor,
                                             render=render)

    def compute_output_loss(self, raw_preds, data: TrainingData, door_balance, balance_weight: float,
                            save_dist_weight: float,
                            graph_diam_weight: float, mc_dist_weight: float, toilet_weight: float):
        all_binary_outputs = torch.cat([data.door_connects, data.missing_connects], dim=1)

        preds = self.get_preds(raw_preds)

        state_value_raw_logodds = torch.cat([preds.door_connects, preds.missing_connects], dim=1)
        # print("train idx: ", num_binary_outputs + num_save_dist_outputs, "num_binary_outputs =", num_binary_outputs, "num_save_dist_outputs =", num_save_dist_outputs)

        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(state_value_raw_logodds,
                                                                           all_binary_outputs.to(
                                                                               state_value_raw_logodds.dtype))

        balance_mask = ~torch.isnan(door_balance)
        balance_zeros = torch.zeros_like(door_balance)
        balance_data0 = torch.where(balance_mask, door_balance, balance_zeros)
        balance_pred0 = torch.where(balance_mask, preds.door_balance, balance_zeros)
        # balance_loss = torch.mean((balance_data0 - balance_pred0) ** 2)
        balance_data0_probs = torch.sigmoid(balance_data0)
        balance_loss = torch.nn.functional.binary_cross_entropy_with_logits(balance_pred0, balance_data0_probs) - \
                       torch.nn.functional.binary_cross_entropy_with_logits(balance_data0, balance_data0_probs)

        toilet_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.toilet_good,
                                                                           data.toilet_good.to(preds.toilet_good.dtype))

        save_dist_mask = (data.save_distances != 255)
        save_dist_loss = torch.mean(
            torch.where(save_dist_mask, (preds.save_dist - data.save_distances.to(torch.float)) ** 2,
                        torch.zeros_like(preds.save_dist)))

        graph_diam_loss = torch.mean((preds.graph_diam - data.graph_diameter.to(torch.float)) ** 2)

        mc_dist_mask = (data.mc_distances != 255)
        mc_dist_loss = torch.mean(torch.where(mc_dist_mask, (preds.mc_dist - data.mc_distances.to(torch.float)) ** 2,
                                              torch.zeros_like(preds.mc_dist)))

        loss = binary_loss + balance_loss * balance_weight + save_dist_loss * save_dist_weight + graph_diam_loss * graph_diam_weight + mc_dist_loss * mc_dist_weight + toilet_loss * toilet_weight
        return loss, binary_loss.item(), balance_loss.item(), save_dist_loss.item(), graph_diam_loss.item(), mc_dist_loss.item(), toilet_loss.item()

    def compute_soft_loss(self, raw_preds, raw_targets, balance_weight: float, save_dist_weight: float,
                          graph_diam_weight: float, mc_dist_weight: float, toilet_weight: float):
        preds = self.get_preds(raw_preds)
        targets = self.get_preds(raw_targets)

        binary_targets_logodds = torch.cat([targets.door_connects, targets.missing_connects], dim=1)
        binary_targets_probs = torch.sigmoid(binary_targets_logodds)
        binary_preds_logodds = torch.cat([preds.door_connects, preds.missing_connects], dim=1)
        # print("train idx: ", num_binary_outputs + num_save_dist_outputs, "num_binary_outputs =", num_binary_outputs, "num_save_dist_outputs =", num_save_dist_outputs)

        binary_loss = \
            torch.nn.functional.binary_cross_entropy_with_logits(binary_preds_logodds, binary_targets_probs) - \
            torch.nn.functional.binary_cross_entropy_with_logits(binary_targets_logodds, binary_targets_probs)

        target_door_balance_probs = torch.sigmoid(targets.door_balance)
        balance_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds.door_balance,
                                                                            target_door_balance_probs) \
                       - torch.nn.functional.binary_cross_entropy_with_logits(targets.door_balance,
                                                                              target_door_balance_probs)

        toilet_target_probs = torch.sigmoid(targets.toilet_good)
        toilet_loss = \
            torch.nn.functional.binary_cross_entropy_with_logits(preds.toilet_good, toilet_target_probs) - \
            torch.nn.functional.binary_cross_entropy_with_logits(targets.toilet_good, toilet_target_probs)

        save_dist_loss = torch.mean(torch.square(preds.save_dist - targets.save_dist))

        graph_diam_loss = torch.mean(torch.square(preds.graph_diam - targets.graph_diam))

        mc_dist_loss = torch.mean(torch.square(preds.mc_dist - targets.mc_dist))

        loss = binary_loss + balance_loss * balance_weight + save_dist_loss * save_dist_weight + graph_diam_loss * graph_diam_weight + mc_dist_loss * mc_dist_weight + toilet_loss * toilet_weight
        return loss, binary_loss.item(), balance_loss.item(), save_dist_loss.item(), graph_diam_loss.item(), mc_dist_loss.item(), toilet_loss.item()

    def compute_losses(self, model, data: TrainingData, balance_weight: float, save_dist_weight: float,
                       graph_diam_weight: float, mc_dist_weight: float, toilet_weight: float,
                       compute_state_value: bool):
        env = self.envs[0]
        # map = env.compute_map(data.room_mask, data.room_position_x, data.room_position_y)
        n = data.room_mask.shape[0]
        device = data.room_mask.device
        action_env_id = torch.arange(data.room_mask.shape[0], device=device)

        preds = model.forward_multiclass(
            data.room_mask, data.room_position_x, data.room_position_y,
            data.map_door_id, action_env_id, data.room_door_id,
            data.steps_remaining, data.round_frac,
            data.temperature, data.mc_dist_coef, env, compute_state_value)

        losses = self.compute_output_loss(preds, data, balance_weight, save_dist_weight,
                                          graph_diam_weight, mc_dist_weight, toilet_weight)
        return losses

    def train_balance(self, data: TrainingData, balance_input):
        env = self.envs[0]
        n = data.room_mask.shape[0]

        # Generate door balance estimates (frequency of each pair of possible door connections, conditioned
        # on temperature and other global parameters that control map generation behavior):
        balance_preds = self.balance_model(balance_input)

        # Update balance model
        door_connect_data = env.get_door_connect_data(data.final_room_mask, data.final_room_position_x,
                                                      data.final_room_position_y)
        balance_labels = torch.cat([
            door_connect_data[0].view(n, -1),
            door_connect_data[1].view(n, -1),
        ], dim=1)
        # balance_loss = torch.mean(torch.square(balance_preds - balance_labels))
        balance_loss = torch.nn.functional.binary_cross_entropy_with_logits(balance_preds, balance_labels)
        self.balance_optimizer.zero_grad()
        balance_loss.backward()
        self.balance_optimizer.step()
        self.balance_average_parameters.update(self.balance_model.parameters())
        return balance_loss

    def train_batch(self, data: TrainingData, state_weight: float, balance_weight: float,
                    save_dist_weight: float, graph_diam_weight: float, mc_dist_weight: float, toilet_weight: float):
        env = self.envs[0]
        device = data.room_mask.device
        n = data.room_mask.shape[0]
        num_left_doors = env.room_left.shape[0]
        num_up_doors = env.room_up.shape[0]

        balance_input = torch.cat([
            torch.log(data.temperature.view(-1, 1)),
            # data.round_frac.view(-1, 1),
            data.mc_dist_coef.view(-1, 1)
        ], dim=1)
        balance_loss = self.train_balance(data, balance_input)

        with torch.no_grad():
            with self.balance_average_parameters.average_parameters(self.balance_model.parameters()):
                balance_preds = self.balance_model(balance_input)
                balance_preds_left_right = balance_preds[:, :(num_left_doors * num_left_doors)].view(n, num_left_doors,
                                                                                                     num_left_doors)
                balance_preds_down_up = balance_preds[:, (num_left_doors * num_left_doors):].view(n, num_up_doors,
                                                                                                  num_up_doors)

        # Get balance scores for the door connections that actually occurred:
        door_balance = env.get_door_balance(
            data.final_room_mask, data.final_room_position_x, data.final_room_position_y,
            balance_preds_left_right, balance_preds_down_up).detach()

        self.action_model.train()
        action_env_id = torch.arange(data.room_mask.shape[0], device=device)
        if state_weight > 0.0:
            aux_state_preds, action_preds = self.action_model.forward_multiclass(
                data.room_mask, data.room_position_x, data.room_position_y,
                data.map_door_id, action_env_id, data.room_door_id,
                data.steps_remaining, data.round_frac,
                data.temperature, data.mc_dist_coef, env, compute_state_value=True)

            aux_state_losses = self.compute_output_loss(aux_state_preds, data, door_balance, balance_weight,
                                                        save_dist_weight, graph_diam_weight,
                                                        mc_dist_weight, toilet_weight)
        else:
            action_preds = self.action_model.forward_multiclass(
                data.room_mask, data.room_position_x, data.room_position_y,
                data.map_door_id, action_env_id, data.room_door_id,
                data.steps_remaining, data.round_frac,
                data.temperature, data.mc_dist_coef, env, compute_state_value=False)

            aux_state_losses = [0.0]

        # Compute action-value losses against hard labels
        action_losses = self.compute_output_loss(action_preds, data, door_balance, balance_weight, save_dist_weight,
                                                 graph_diam_weight,
                                                 mc_dist_weight, toilet_weight)

        self.action_optimizer.zero_grad()
        self.action_grad_scaler.scale(action_losses[0] + state_weight * aux_state_losses[0]).backward()
        self.action_grad_scaler.step(self.action_optimizer)
        self.action_grad_scaler.update()
        self.action_model.decay(self.decay_amount * self.action_optimizer.param_groups[0]['lr'])
        self.action_model.project()
        self.action_average_parameters.update(self.action_model.all_param_data())

        return aux_state_losses, balance_loss, action_losses

    def eval_batch(self, data: TrainingData, next_data: TrainingData, balance_weight: float, save_dist_weight: float,
                   graph_diam_weight: float, mc_dist_weight: float, toilet_weight: float):
        self.state_model.eval()
        self.action_model.eval()
        env = self.envs[0]
        with torch.no_grad():
            device = data.room_mask.device
            action_env_id = torch.arange(data.room_mask.shape[0], device=device)

            state_preds, action_preds = self.action_model.forward_multiclass(
                data.room_mask, data.room_position_x, data.room_position_y,
                data.map_door_id, action_env_id, data.room_door_id,
                data.steps_remaining, data.round_frac,
                data.temperature, data.mc_dist_coef, env, compute_state_value=True)

            state_losses = self.compute_output_loss(state_preds, data, balance_weight, save_dist_weight,
                                                    graph_diam_weight,
                                                    mc_dist_weight, toilet_weight)

            action_losses = self.compute_output_loss(action_preds, data, balance_weight, save_dist_weight,
                                                     graph_diam_weight,
                                                     mc_dist_weight, toilet_weight)

            next_preds, _ = self.action_model.forward_multiclass(
                next_data.room_mask, next_data.room_position_x, next_data.room_position_y,
                next_data.map_door_id, action_env_id, next_data.room_door_id,
                next_data.steps_remaining, next_data.round_frac,
                next_data.temperature, next_data.mc_dist_coef, env, compute_state_value=True)

            next_losses = self.compute_output_loss(next_preds, next_data, balance_weight, save_dist_weight,
                                                   graph_diam_weight,
                                                   mc_dist_weight, toilet_weight)

        return state_losses, action_losses, next_losses
