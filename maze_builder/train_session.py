import torch
from maze_builder.model import Network
from maze_builder.env import MazeBuilderEnv
from model_average import ExponentialAverage
import logging

# TODO: look at using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 network: Network,
                 optimizer: torch.optim.Optimizer,
                 ema_beta: float,
                 huber_delta: float,
                 ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        # self.average_parameters = SimpleAverage(network.all_param_data())
        self.average_parameters = ExponentialAverage(network.all_param_data(), beta=ema_beta)
        self.num_rounds = 0
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.loss_obj = torch.nn.HuberLoss(delta=huber_delta)

        self.total_step_remaining_gen = 0.0
        self.total_step_remaining_train = 0.0


    def forward_state_action(self, room_mask, room_position_x, room_position_y, action_candidates, steps_remaining, gen=True):
        # torch.cuda.synchronize()
        # logging.info("Processing candidate data")
        num_envs = room_mask.shape[0]
        num_candidates = action_candidates.shape[1]
        num_rooms = len(self.env.rooms)
        action_room_id = action_candidates[:, :, 0]
        action_x = action_candidates[:, :, 1]
        action_y = action_candidates[:, :, 2]
        all_room_mask = room_mask.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_x = room_position_x.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_room_position_y = room_position_y.unsqueeze(1).repeat(1, num_candidates + 1, 1)
        all_steps_remaining = steps_remaining.unsqueeze(1).repeat(1, num_candidates + 1)

        # print(action_candidates.device, action_room_id.device)
        all_room_mask[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                      torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                      action_room_id] = True
        all_room_mask[:, :, -1] = False
        all_room_position_x[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_x
        all_room_position_y[torch.arange(num_envs, device=action_candidates.device).view(-1, 1),
                            torch.arange(1, 1 + num_candidates, device=action_candidates.device).view(1, -1),
                            action_room_id] = action_y
        all_steps_remaining[:, 1:] -= 1

        room_mask_flat = all_room_mask.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_x_flat = all_room_position_x.view(num_envs * (1 + num_candidates), num_rooms)
        room_position_y_flat = all_room_position_y.view(num_envs * (1 + num_candidates), num_rooms)
        steps_remaining_flat = all_steps_remaining.view(num_envs * (1 + num_candidates))

        # torch.cuda.synchronize()
        # logging.info("Creating map")

        map_flat = self.env.compute_map(room_mask_flat, room_position_x_flat, room_position_y_flat)

        # torch.cuda.synchronize()
        # logging.info("Model forward")
        out_flat = self.network(map_flat, room_mask_flat, steps_remaining_flat)
        torch.cuda.synchronize()
        out = out_flat.view(num_envs, 1 + num_candidates)
        state_value = out[:, 0]
        action_value = out[:, 1:]
        return state_value, action_value

    def generate_episode(self, episode_length: int, num_candidates: int, temperature: float, explore_eps: float,
                       render=False):
        device = self.env.device
        self.env.reset()
        room_mask_list = []
        room_position_x_list = []
        room_position_y_list = []
        state_value_list = []
        action_value_list = []
        action_list = []
        self.network.eval()
        # torch.cuda.synchronize()
        # logging.debug("Averaging parameters")
        with self.average_parameters.average_parameters(self.network.all_param_data()):
            total_action_prob = 0.0
            for j in range(episode_length):
                if render:
                    self.env.render()
                # torch.cuda.synchronize()
                # logging.debug("Getting candidates")
                action_candidates = self.env.get_action_candidates(num_candidates)
                steps_remaining = torch.full([self.env.num_envs], episode_length - j,
                                             dtype=torch.float32, device=device)
                room_mask = self.env.room_mask.clone()
                room_position_x = self.env.room_position_x.clone()
                room_position_y = self.env.room_position_y.clone()
                with torch.no_grad():
                    state_value, action_value = self.forward_state_action(
                        self.env.room_mask, self.env.room_position_x, self.env.room_position_y,
                        action_candidates, steps_remaining)
                action_probs = torch.softmax(action_value * temperature, dim=1)
                action_probs = torch.full_like(action_probs, explore_eps / num_candidates) + (
                            1 - explore_eps) * action_probs
                action_index = _rand_choice(action_probs)
                selected_action_prob = action_probs[torch.arange(self.env.num_envs, device=device), action_index]
                action = action_candidates[torch.arange(self.env.num_envs, device=device), action_index, :]
                selected_action_value = action_value[torch.arange(self.env.num_envs, device=device), action_index]

                self.env.step(action)
                room_mask_list.append(room_mask)
                room_position_x_list.append(room_position_x)
                room_position_y_list.append(room_position_y)
                action_list.append(action)
                state_value_list.append(state_value)
                action_value_list.append(selected_action_value)
                total_action_prob += torch.mean(selected_action_prob).item()
            room_mask_tensor = torch.stack(room_mask_list, dim=0)
            room_position_x_tensor = torch.stack(room_position_x_list, dim=0)
            room_position_y_tensor = torch.stack(room_position_y_list, dim=0)
            state_value_tensor = torch.stack(state_value_list, dim=0)
            action_value_tensor = torch.stack(action_value_list, dim=0)
            action_tensor = torch.stack(action_list, dim=0)
            reward_tensor = self.env.reward()
            action_prob = total_action_prob / episode_length
        return room_mask_tensor, room_position_x_tensor, room_position_y_tensor, state_value_tensor, \
               action_value_tensor, action_tensor, reward_tensor, action_prob

    def generate_round(self, num_episodes, episode_length: int, num_candidates: int, temperature: float, explore_eps: float,
                         render=False):

        # logging.debug("Generating data")
        room_mask_list = []
        room_position_x_list = []
        room_position_y_list = []
        state_value_list = []
        action_value_list = []
        action_list = []
        reward_list = []
        action_prob_total = 0.0
        for _ in range(num_episodes):
            room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = self.generate_episode(
                    episode_length=episode_length,
                    num_candidates=num_candidates,
                    temperature=temperature,
                    explore_eps=explore_eps,
                    render=render)
            room_mask_list.append(room_mask)
            room_position_x_list.append(room_position_x)
            room_position_y_list.append(room_position_y)
            state_value_list.append(state_value)
            action_value_list.append(action_value)
            action_list.append(action)
            reward_list.append(reward)
            action_prob_total += action_prob
        room_mask = torch.cat(room_mask_list, dim=1)
        room_position_x = torch.cat(room_position_x_list, dim=1)
        room_position_y = torch.cat(room_position_y_list, dim=1)
        state_value = torch.cat(state_value_list, dim=1)
        action_value = torch.cat(action_value_list, dim=1)
        action = torch.cat(action_list, dim=1)
        reward = torch.cat(reward_list, dim=0)
        action_prob = action_prob_total / num_episodes
        return room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob

    def train_round(self,
                    num_rounds: int,
                    episode_length: int,
                    batch_size: int,
                    num_candidates: int,
                    temperature: float,
                    num_passes: int = 1,
                    td_lambda: float = 0.0,
                    explore_eps: float = 0.0,
                    lr_decay: float = 1.0,
                    render: bool = False,
                    dry_run: bool = False,
                    ):
        self.map_rand = None

        num_episodes = self.env.num_envs * num_rounds
        room_mask, room_position_x, room_position_y, state_value, action_value, action, reward, action_prob = self.generate_round(
            num_episodes=num_rounds,
            episode_length=episode_length,
            num_candidates=num_candidates,
            temperature=temperature,
            explore_eps=explore_eps,
            render=render)

        # torch.cuda.synchronize()
        # logging.debug("Computing metrics and targets")
        steps_remaining = (episode_length - torch.arange(episode_length, device=self.env.device)).view(-1, 1).repeat(1, num_episodes)

        mean_reward = torch.mean(reward.to(torch.float32))
        max_reward = torch.max(reward).item()
        cnt_max_reward = torch.sum(reward == max_reward)

        turn_pass = action[:, :, 0] == len(self.env.rooms) - 1
        all_pass = torch.flip(torch.cummin(torch.flip(turn_pass, dims=[0]), dim=0)[0], dims=[0])
        frac_pass = torch.mean((turn_pass & ~all_pass).to(torch.float32))

        # Compute Monte-Carlo error
        mc_err = torch.mean((state_value - reward.unsqueeze(0)) ** 2).item()
        mc_bias = torch.mean(state_value - reward.unsqueeze(0)).item()

        # Compute the TD targets
        target_list = []
        target_batch = reward.to(torch.float32)
        target_list.append(reward)
        for i in reversed(range(1, episode_length)):
            state_value1 = state_value[i, :]
            target_batch = td_lambda * target_batch + (1 - td_lambda) * state_value1
            target_list.append(target_batch)
        target = torch.stack(list(reversed(target_list)), dim=0)

        # Flatten the data
        n = episode_length * num_episodes
        room_mask = room_mask.view(n, len(self.env.rooms))
        room_position_x = room_position_x.view(n, len(self.env.rooms))
        room_position_y = room_position_y.view(n, len(self.env.rooms))
        action = action.view(n, 3)
        steps_remaining = steps_remaining.view(n)
        target = target.view(n)
        # all_pass = all_pass.view(n)

        # # Filter out completed game states (this would need to be modified to be made correct with bootstrapping)
        # keep = ~all_pass
        # map0 = map0[keep]
        # room_mask = room_mask[keep]
        # action = action[keep]
        # steps_remaining = steps_remaining[keep]
        # target = target[keep]
        # n = map0.shape[0]

        # Shuffle the data
        # torch.cuda.synchronize()
        # logging.debug("Shuffling")
        perm = torch.randperm(n)
        room_mask = room_mask[perm, :]
        room_position_x = room_position_x[perm, :]
        room_position_y = room_position_y[perm, :]
        action = action[perm]
        steps_remaining = steps_remaining[perm]
        target = target[perm]

        num_batches = n // batch_size

        lr_decay_per_step = lr_decay ** (1 / num_passes / num_batches)

        total_loss = 0.0
        total_err = 0.0
        total_cnt = 0
        for _ in range(num_passes):
            self.network.train()
            # self.average_parameters.reset()
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                room_mask_batch = room_mask[start:end, :]
                room_position_x_batch = room_position_x[start:end, :]
                room_position_y_batch = room_position_y[start:end, :]
                steps_remaining_batch = steps_remaining[start:end]
                # action_batch = action[start:end, :]
                target_batch = target[start:end]

                if dry_run:
                    with torch.no_grad():
                        state_value0, _ = self.forward_state_action(
                            room_mask_batch, room_position_x_batch, room_position_y_batch,
                            torch.zeros([batch_size, 0, 3], dtype=torch.int64, device=room_mask_batch.device),
                            steps_remaining_batch,
                            gen=False)
                else:
                    state_value0, _ = self.forward_state_action(
                        room_mask_batch, room_position_x_batch, room_position_y_batch,
                        torch.zeros([batch_size, 0, 3], dtype=torch.int64, device=room_mask_batch.device),
                        steps_remaining_batch,
                        gen=False)

                err = state_value0 - target_batch
                # loss = torch.mean(err ** 2)
                # loss = torch.mean(torch.abs(err))
                loss = self.loss_obj(state_value0, target_batch)
                total_cnt += err.shape[0]
                # print(state_loss, action_loss)
                # loss = (1 - action_loss_weight) * state_loss + action_loss_weight * action_loss
                self.optimizer.zero_grad()
                if not dry_run:
                    # torch.cuda.synchronize()
                    # logging.debug("Model backward")
                    self.grad_scaler.scale(loss).backward()

                    # torch.cuda.synchronize()
                    # logging.debug("Optimizer step")
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                    # loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1e-5)
                    # self.optimizer.step()
                self.optimizer.param_groups[0]['lr'] *= lr_decay_per_step
                self.average_parameters.update(self.network.all_param_data())
                # # self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
                total_loss += loss.item()
                total_err += torch.mean(err).item()

        self.num_rounds += 1

        # total_loss = 0
        # num_batches = 1
        # total_mc_err = 0
        return mean_reward, max_reward, cnt_max_reward, total_loss / num_batches / num_passes, total_err / num_passes / num_batches, \
               mc_err, mc_bias, action_prob, frac_pass

