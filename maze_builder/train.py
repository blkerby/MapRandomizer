# TODO:
#  - try implementing DQN, architecture similar to dueling network:
#    - single network with two heads: one for state-value and one for action-value (or action-advantages)
#    - target for both state-values and action-values is the estimate state-value n steps later,
#      - for stability/accuracy, target computed using an averaged version of the network (EMA or simple average
#        from the last round)
#  - noisy nets: for strategic/coordinated exploration?
#      - instead of randomizing weights/biases, maybe just add noise (with tunable scale) to activations
#        in certain layer(s) (same noise across all time steps of an episode)
#  - distributional DQN: split space of rewards into buckets and predict probabilities
#  - prioritized replay
#  - try some of the new ideas on Atari benchmarks (variation of dueling network, and variation of noisy nets)
import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import Room
import logic.rooms.crateria
from datetime import datetime
from typing import List, Optional
import pickle
from model_average import SimpleAverage

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
torch.autograd.set_detect_anomaly(True)

start_time = datetime.now()
pickle_name = 'models/crateria-{}.pkl'.format(start_time.isoformat())
logging.info("Checkpoint path: {}".format(pickle_name))


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class GlobalMaxPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.max(X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]), dim=2)[0]


# TODO: look at using torch.multinomial instead of implementing this from scratch?
def _rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class Network(torch.nn.Module):
    def __init__(self, room_tensor, map_x, map_y, map_channels, map_kernel_size, fc_widths):
        super().__init__()
        self.room_tensor = room_tensor

        map_layers = []
        map_channels = [5] + map_channels
        width = map_x
        height = map_y
        for i in range(len(map_channels) - 1):
            map_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                              kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                              padding=map_kernel_size[i] // 2))
            map_layers.append(torch.nn.ReLU())
            # map_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
            map_layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
            width = (width + 1) // 2
            height = (height + 1) // 2
        # map_layers.append(GlobalAvgPool2d())
        # map_layers.append(GlobalMaxPool2d())
        map_layers.append(torch.nn.Flatten())
        self.map_sequential = torch.nn.Sequential(*map_layers)

        fc_layers = []
        fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + fc_widths
        for i in range(len(fc_widths) - 1):
            fc_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            fc_layers.append(torch.nn.ReLU())
            # fc_layers.append(torch.nn.BatchNorm1d(fc_widths[i + 1], momentum=batch_norm_momentum))
        fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.fc_sequential = torch.nn.Sequential(*fc_layers)

    def forward(self, map, room_mask, candidate_placements, steps_remaining):
        X = map.to(torch.float32)
        for layer in self.map_sequential:
            # print(X.shape, layer)
            X = layer(X)

        X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
        for layer in self.fc_sequential:
            X = layer(X)

        state_value = X[:, 0]

        # TODO: actually compute something here:
        action_value = torch.zeros([candidate_placements.shape[0], candidate_placements.shape[1]], dtype=torch.float32,
                                   device=map.device)

        return state_value, action_value


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 network: Network,
                 optimizer: torch.optim.Optimizer,
                 ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.average_parameters = SimpleAverage(
            network.parameters())  # TODO: try batch norm again, and add its internal tensors here
        self.num_rounds = 0

    def generate_round(self, episode_length: int, num_candidates: int, temperature: float, render=False):
        device = self.env.map.device
        map, room_mask = self.env.reset()
        map_list = [map]
        room_mask_list = [room_mask]
        action_list = []
        self.network.eval()
        # with self.average_parameters.average_parameters():
        for j in range(episode_length):
            if render:
                self.env.render()
            candidate_placements = env.get_placement_candidates(num_candidates)
            steps_remaining = torch.full([self.env.num_envs], episode_length - j,
                                         dtype=torch.float32, device=device)
            with torch.no_grad():
                state_value, action_value = self.network(map, room_mask, candidate_placements, steps_remaining)
            action_probs = torch.softmax(action_value * temperature, dim=1)
            action_index = _rand_choice(action_probs)
            action = candidate_placements[torch.arange(self.env.num_envs, device=device), action_index]
            map, room_mask = self.env.step(action[:, 0], action[:, 1], action[:, 2])
            map_list.append(map)
            room_mask_list.append(room_mask)
            action_list.append(action)
        map_tensor = torch.stack(map_list, dim=0)
        room_mask_tensor = torch.stack(room_mask_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        reward_tensor = self.env.reward()
        return map_tensor, room_mask_tensor, action_tensor, reward_tensor

    def train_round(self,
                    episode_length: int,
                    batch_size: int,
                    td_lambda: float = 0.0,
                    policy_variation_penalty: float = 0.0,
                    # mc_weight: float = 0.0,
                    render: bool = False,
                    ):
        # Generate data using the current policy
        map, room_mask, position, direction, action, reward = self.generate_round(episode_length=episode_length,
                                                                                  render=render)

        cumul_reward = torch.flip(torch.cumsum(torch.flip(reward, dims=[0]), dim=0), dims=[0])
        map0 = map[:-1, :, :, :, :]
        map1 = map[1:, :, :, :, :]
        room_mask0 = room_mask[:-1, :, :]
        room_mask1 = room_mask[1:, :, :]
        steps_remaining = (episode_length - torch.arange(episode_length, device=map.device)).view(-1, 1).repeat(1,
                                                                                                                env.num_envs)
        total_reward = cumul_reward[0, :]
        mean_reward = torch.mean(total_reward.to(torch.float32))
        max_reward = torch.max(total_reward).item()
        cnt_max_reward = torch.sum(total_reward == max_reward)

        # Compute the TD targets
        with self.average_value_parameters.average_parameters():
            target_list = []
            total_target_err = 0.0
            self.value_network.eval()
            target_batch = reward[-1, :]
            target_list.append(target_batch)
            for i in reversed(range(episode_length - 1)):
                map1_batch = map1[i, :, :, :, :]
                room_mask1_batch = room_mask1[i, :, :]
                reward_batch = reward[i, :]
                cumul_reward_batch = cumul_reward[i, :]
                steps_remaining_batch = steps_remaining[i, :]
                with torch.no_grad():
                    value1 = self.value_network(map1_batch, room_mask1_batch, steps_remaining_batch - 1)
                # target_batch = torch.where(steps_remaining_batch == 1, reward_batch.to(torch.float32), reward_batch + value1)
                target_batch = td_lambda * target_batch + (1 - td_lambda) * value1 + reward_batch
                target_list.append(target_batch)
                total_target_err += torch.mean((value1 - cumul_reward_batch) ** 2).item()
            target = torch.stack(list(reversed(target_list)), dim=0)

        # Flatten the data
        n = episode_length * self.env.num_envs
        map0 = map0.view(n, 3, self.env.map_x, self.env.map_y)
        # map1 = map1.view(n, 3, self.env.map_x, self.env.map_y)
        room_mask0 = room_mask0.view(n, len(self.env.rooms))
        # room_mask1 = room_mask1.view(n, len(self.env.rooms))
        position = position.view(n, 2)
        direction = direction.view(n)
        action = action.view(n)
        # reward = reward.view(n)
        cumul_reward = cumul_reward.view(n)
        steps_remaining = steps_remaining.view(n)
        target = target.view(n)

        # Shuffle the data
        perm = torch.randperm(n)
        map0 = map0[perm, :, :, :]
        # map1 = map1[perm, :, :, :]
        room_mask0 = room_mask0[perm, :]
        # room_mask1 = room_mask1[perm, :]
        position = position[perm]
        direction = direction[perm]
        action = action[perm]
        # reward = reward[perm]
        # cumul_reward = cumul_reward[perm]
        steps_remaining = steps_remaining[perm]
        target = target[perm]

        num_batches = n // batch_size

        # Make one pass through the data, updating the value network
        total_value_loss_bs = 0.0
        total_policy_loss = 0.0
        total_policy_variation = 0.0
        self.value_network.train()
        self.policy_network.train()
        self.average_value_parameters.reset()
        self.average_policy_parameters.reset()
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            map0_batch = map0[start:end, :, :, :]
            # map1_batch = map1[start:end, :, :, :]
            room_mask0_batch = room_mask0[start:end, :]
            # room_mask1_batch = room_mask1[start:end, :]
            # reward_batch = reward[start:end]
            # cumul_reward_batch = cumul_reward[start:end]
            steps_remaining_batch = steps_remaining[start:end]
            action_batch = action[start:end]
            position_batch = position[start:end, :]
            direction_batch = direction[start:end]
            target_batch = target[start:end]

            # Update the value network
            value0 = self.value_network(map0_batch, room_mask0_batch, steps_remaining_batch)
            # with torch.no_grad():
            #     value1 = self.value_network(map1_batch, room_mask1_batch, steps_remaining_batch - 1)
            #     target = torch.where(steps_remaining_batch == 1, reward_batch.to(torch.float32), reward_batch + value1)
            # target = targets[i]
            value_loss_bs = torch.mean((value0 - target_batch) ** 2)
            # value_loss_mc = torch.mean((value0 - cumul_reward_batch) ** 2)
            # value_loss = (1 - mc_weight) * value_loss_bs + mc_weight * value_loss_mc
            self.value_optimizer.zero_grad()
            value_loss_bs.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1e-5)
            self.value_optimizer.step()
            self.average_value_parameters.update()
            # self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
            total_value_loss_bs += value_loss_bs.item()

            # Update the policy network
            # advantage = (1 - mc_weight) * target_batch + mc_weight * cumul_reward_batch - value0.detach()
            advantage = target_batch - value0.detach()
            left_ids = torch.nonzero(direction_batch == 0)[:, 0]
            right_ids = torch.nonzero(direction_batch == 1)[:, 0]
            down_ids = torch.nonzero(direction_batch == 2)[:, 0]
            up_ids = torch.nonzero(direction_batch == 3)[:, 0]
            policy_out = self.policy_network(map0_batch, room_mask0_batch, position_batch, direction_batch,
                                             left_ids, right_ids, down_ids, up_ids, steps_remaining_batch)
            left_raw_logprobs, right_raw_logprobs, down_raw_logprobs, up_raw_logprobs = policy_out
            left_logprobs = left_raw_logprobs - torch.logsumexp(left_raw_logprobs, dim=1, keepdim=True)
            right_logprobs = right_raw_logprobs - torch.logsumexp(right_raw_logprobs, dim=1, keepdim=True)
            down_logprobs = down_raw_logprobs - torch.logsumexp(down_raw_logprobs, dim=1, keepdim=True)
            up_logprobs = up_raw_logprobs - torch.logsumexp(up_raw_logprobs, dim=1, keepdim=True)
            left_logprobs_action = left_logprobs[torch.arange(len(left_ids), device=device), action_batch[left_ids]]
            right_logprobs_action = right_logprobs[torch.arange(len(right_ids), device=device), action_batch[right_ids]]
            down_logprobs_action = down_logprobs[torch.arange(len(down_ids), device=device), action_batch[down_ids]]
            up_logprobs_action = up_logprobs[torch.arange(len(up_ids), device=device), action_batch[up_ids]]
            policy_loss = -(torch.sum(advantage[left_ids] * left_logprobs_action) +
                            torch.sum(advantage[right_ids] * right_logprobs_action) +
                            torch.sum(advantage[down_ids] * down_logprobs_action) +
                            torch.sum(advantage[up_ids] * up_logprobs_action)) / batch_size
            num_doors = (self.env.left_door_tensor.shape[0] + self.env.right_door_tensor.shape[0] +
                         self.env.down_door_tensor.shape[0] + self.env.up_door_tensor.shape[0])
            policy_variation = (torch.sum(left_raw_logprobs ** 2) +
                                torch.sum(right_raw_logprobs ** 2) +
                                torch.sum(down_raw_logprobs ** 2) +
                                torch.sum(up_raw_logprobs ** 2)) / batch_size / num_doors
            policy_variation_loss = policy_variation_penalty * policy_variation
            self.policy_optimizer.zero_grad()
            (policy_loss + policy_variation_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1e-5)
            self.policy_optimizer.step()
            self.average_policy_parameters.update()
            # self.policy_network.decay(weight_decay * self.policy_optimizer.param_groups[0]['lr'])
            total_policy_loss += policy_loss.item()
            total_policy_variation += policy_variation.item()

        self.num_rounds += 1

        return mean_reward, max_reward, cnt_max_reward, total_value_loss_bs / num_batches, \
               total_target_err / episode_length, total_policy_loss / num_batches, total_policy_variation / num_batches


import logic.rooms.crateria
import logic.rooms.crateria_isolated
import logic.rooms.wrecked_ship
import logic.rooms.norfair_lower
import logic.rooms.norfair_upper
import logic.rooms.all_rooms
import logic.rooms.brinstar_pink
import logic.rooms.brinstar_green
import logic.rooms.brinstar_red
import logic.rooms.brinstar_blue
import logic.rooms.maridia_lower
import logic.rooms.maridia_upper

device = torch.device('cpu')
# device = torch.device('cuda:0')

# num_envs = 256
num_envs = 32
rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.crateria.rooms
# rooms = logic.rooms.crateria.rooms + logic.rooms.wrecked_ship.rooms
# rooms = logic.rooms.wrecked_ship.rooms
# rooms = logic.rooms.norfair_lower.rooms + logic.rooms.norfair_upper.rooms
# rooms = logic.rooms.norfair_upper.rooms
# rooms = logic.rooms.norfair_lower.rooms
# rooms = logic.rooms.brinstar_warehouse.rooms
# rooms = logic.rooms.brinstar_pink.rooms
# rooms = logic.rooms.brinstar_red.rooms
# rooms = logic.rooms.brinstar_blue.rooms
# rooms = logic.rooms.brinstar_green.rooms
# rooms = logic.rooms.maridia_lower.rooms
# rooms = logic.rooms.maridia_upper.rooms
# rooms = logic.rooms.all_rooms.rooms
episode_length = 64
display_freq = 1
map_x = 40
map_y = 30
# map_x = 10
# map_y = 10
env = MazeBuilderEnv(rooms,
                     map_x=map_x,
                     map_y=map_y,
                     max_room_width=11,
                     num_envs=num_envs,
                     device=device)
# print(env.room_tensor.shape, env.left_door_tensor.shape, env.right_door_tensor.shape, env.down_door_tensor.shape, env.up_door_tensor.shape)


network = Network(env.room_tensor,
                  map_x=env.padded_map_x,
                  map_y=env.padded_map_y,
                  map_channels=[32, 64, 128],
                  map_kernel_size=[11, 9, 5],
                  fc_widths=[128, 128, 128],
                  ).to(device)
network.fc_sequential[-1].weight.data[:, :] = 0.0
network.fc_sequential[-1].bias.data[:] = 0.0
optimizer = torch.optim.Adam(network.parameters(), lr=0.0005, betas=(0.5, 0.5), eps=1e-15)

print(network)
print(optimizer)
logging.info("Starting training")

session = TrainingSession(env,
                          network=network,
                          optimizer=optimizer)

torch.set_printoptions(linewidth=120, threshold=10000)

num_candidates = 16
temperature = 1.0
map_tensor, room_mask_tensor, action_tensor, reward_tensor = session.generate_round(episode_length, num_candidates,
                                                                                    temperature)

# # map_tensor, room_mask_tensor, position_tensor, direction_tensor, action_tensor, reward_tensor = session.generate_round(
# #     episode_length=episode_length,
# #     render=True)
#
# #
# # # session = pickle.load(open('models/crateria-2021-06-29T13:35:06.399214.pkl', 'rb'))
# #
# import io
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name =='_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
# session = CPU_Unpickler(open('models/crateria-2021-07-12T15:28:23.905530.pkl', 'rb')).load()
# # session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
# batch_size = 2 ** 8
# # batch_size = 2 ** 13  # 2 ** 12
# policy_variation_penalty = 0.1
# td_lambda = 0.5
# session.env = env
# # session.value_optimizer.param_groups[0]['lr'] = 0.0001
# # session.policy_optimizer.param_groups[0]['lr'] = 2e-5
# # session.value_optimizer.param_groups[0]['betas'] = (0.5, 0.5)
# # session.policy_optimizer.param_groups[0]['betas'] = (0.5, 0.5)
#
# logging.info(
#     "num_envs={}, batch_size={}, policy_variation_penalty={}, td_lambda={}".format(session.env.num_envs, batch_size,
#                                                                      policy_variation_penalty, td_lambda))
# # for i in range(100000):
# #     mean_reward, max_reward, cnt_max_reward, value_loss_bs, target_err, policy_loss, policy_variation = session.train_round(
# #         episode_length=episode_length,
# #         batch_size=batch_size,
# #         policy_variation_penalty=policy_variation_penalty,
# #         td_lambda=td_lambda,
# #         # mc_weight=0.1,
# #         # render=True)
# #         render=False)
# #     # render=i % display_freq == 0)
# #     logging.info("{}: reward={:.3f} (max={:d}, cnt={:d}), value={:.5f}, target={:.5f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
# #         session.num_rounds, mean_reward, max_reward, cnt_max_reward, value_loss_bs, target_err, policy_loss, policy_variation))
# #     pickle.dump(session, open(pickle_name, 'wb'))
#
#
# while True:
#     map_tensor, room_mask_tensor, position_tensor, direction_tensor, action_tensor, reward_tensor = session.generate_round(episode_length, render=False)
#     sum_reward = torch.sum(reward_tensor, dim=0)
#     max_reward, max_reward_ind = torch.max(sum_reward, dim=0)
#     logging.info("{}: {}".format(max_reward, sum_reward.tolist()))
#     if max_reward.item() >= 33:
#         break
# # session.env.render(0)
# session.env.render(max_reward_ind.item())
