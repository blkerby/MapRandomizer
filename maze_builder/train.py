# TODO:
import torch
import logging
from maze_builder.env import MazeBuilderEnv
from maze_builder.types import Room
import logic.rooms.crateria
from datetime import datetime
from typing import List, Optional
import pickle

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])
torch.autograd.set_detect_anomaly(True)

start_time = datetime.now()
pickle_name = 'models/norfair-{}.pkl'.format(start_time.isoformat())
logging.info("Checkpoint path: {}".format(pickle_name))


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class GlobalMaxPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.max(X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3]), dim=2)[0]


class PolicyNetwork(torch.nn.Module):
    def __init__(self, room_tensor, left_door_tensor, right_door_tensor, down_door_tensor, up_door_tensor,
                 map_x, map_y, map_channels, map_kernel_size, fc_widths, door_embedding_width, batch_norm_momentum):
        super().__init__()
        self.room_tensor = room_tensor
        self.left_door_tensor = left_door_tensor
        self.right_door_tensor = right_door_tensor
        self.down_door_tensor = down_door_tensor
        self.up_door_tensor = up_door_tensor
        # self.local_radius = local_radius

        map_layers = []
        map_channels = [4] + map_channels
        width = map_x
        height = map_y
        for i in range(len(map_channels) - 1):
            map_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                              kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                              padding=map_kernel_size[i] // 2))
            map_layers.append(torch.nn.ReLU())
            map_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
            map_layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
            width = (width + 1) // 2
            height = (height + 1) // 2
        # map_layers.append(GlobalAvgPool2d())
        # map_layers.append(GlobalMaxPool2d())
        map_layers.append(torch.nn.Flatten())
        self.map_sequential = torch.nn.Sequential(*map_layers)

        # map_layers = []
        # map_channels = [3] + map_channels
        # width = 2 * local_radius + 1
        # height = 2 * local_radius + 1
        # for i in range(len(map_channels) - 1):
        #     map_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
        #                                       kernel_size=(map_kernel_size[i], map_kernel_size[i]),
        #                                       padding=map_kernel_size[i] // 2))
        #     map_layers.append(torch.nn.ReLU())
        #     map_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
        #     map_layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
        #     width = (width + 1) // 2
        #     height = (height + 1) // 2
        # # map_layers.append(GlobalAvgPool2d())
        # # map_layers.append(GlobalMaxPool2d())
        # map_layers.append(torch.nn.Flatten())
        # self.map_sequential = torch.nn.Sequential(*map_layers)

        fc_layers = []
        fc_widths = [(width * height * map_channels[-1]) + 1 + room_tensor.shape[0]] + fc_widths
        for i in range(len(fc_widths) - 1):
            fc_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            fc_layers.append(torch.nn.ReLU())
            fc_layers.append(torch.nn.BatchNorm1d(fc_widths[i + 1], momentum=batch_norm_momentum))
        fc_layers.append(torch.nn.Linear(fc_widths[-1], door_embedding_width))
        self.fc_sequential = torch.nn.Sequential(*fc_layers)

        self.left_door_embedding = torch.nn.Parameter(torch.randn([door_embedding_width, left_door_tensor.shape[0]]))
        self.right_door_embedding = torch.nn.Parameter(torch.randn([door_embedding_width, right_door_tensor.shape[0]]))
        self.down_door_embedding = torch.nn.Parameter(torch.randn([door_embedding_width, down_door_tensor.shape[0]]))
        self.up_door_embedding = torch.nn.Parameter(torch.randn([door_embedding_width, up_door_tensor.shape[0]]))

        # self.left_raw_logprobs = torch.nn.Parameter(torch.zeros([self.right_door_tensor.shape[0]], dtype=torch.float32))
        # self.right_raw_logprobs = torch.nn.Parameter(torch.zeros([self.left_door_tensor.shape[0]], dtype=torch.float32))
        # self.down_raw_logprobs = torch.nn.Parameter(torch.zeros([self.up_door_tensor.shape[0]], dtype=torch.float32))
        # self.up_raw_logprobs = torch.nn.Parameter(torch.zeros([self.down_door_tensor.shape[0]], dtype=torch.float32))


    def forward(self, map, room_mask, position, direction, left_ids, right_ids, down_ids, up_ids, steps_remaining):
        device = map.device

        # padded_map = torch.nn.functional.pad(map, pad=(self.local_radius, self.local_radius, self.local_radius, self.local_radius))
        # index_x = torch.arange(-self.local_radius, self.local_radius + 1, device=device).view(1, 1, -1, 1) + position[:, 0].view(-1, 1, 1, 1)
        # index_y = torch.arange(-self.local_radius, self.local_radius + 1, device=device).view(1, 1, 1, -1) + position[:, 1].view(-1, 1, 1, 1)
        # local_map = padded_map[torch.arange(map.shape[0], device=device).view(-1, 1, 1, 1),
        #                        torch.arange(3, device=device).view(1, -1, 1, 1), index_x, index_y]
        # X = local_map
        X = map.to(torch.float32)
        pos_layer = torch.zeros_like(X[:, :1, :, :])
        pos_layer[torch.arange(map.shape[0], device=device), 0, position[:, 0], position[:, 1]] = (direction + 1).to(torch.float32)
        X = torch.cat([X, pos_layer], dim=1)
        for layer in self.map_sequential:
            # print(X.shape, layer)
            X = layer(X)

        X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
        for layer in self.fc_sequential:
            X = layer(X)

        X_left = X[left_ids, :]
        X_right = X[right_ids, :]
        X_down = X[down_ids, :]
        X_up = X[up_ids, :]

        left_raw_logprobs = torch.matmul(X_left, self.right_door_embedding)
        right_raw_logprobs = torch.matmul(X_right, self.left_door_embedding)
        down_raw_logprobs = torch.matmul(X_down, self.up_door_embedding)
        up_raw_logprobs = torch.matmul(X_up, self.down_door_embedding)

        return left_raw_logprobs, right_raw_logprobs, down_raw_logprobs, up_raw_logprobs


class ValueNetwork(torch.nn.Module):
    def __init__(self, room_tensor, map_x, map_y, map_channels, map_kernel_size, fc_widths, batch_norm_momentum):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.room_tensor = room_tensor

        map_layers = []
        map_channels = [3] + map_channels
        width = map_x
        height = map_y
        for i in range(len(map_channels) - 1):
            map_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                              kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                              padding=map_kernel_size[i] // 2))
            map_layers.append(torch.nn.ReLU())
            map_layers.append(torch.nn.BatchNorm2d(map_channels[i + 1], momentum=batch_norm_momentum))
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
            fc_layers.append(torch.nn.BatchNorm1d(fc_widths[i + 1], momentum=batch_norm_momentum))
        fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.fc_sequential = torch.nn.Sequential(*fc_layers)
        # self.lin = torch.nn.Linear(1, 1)
        # self.dummy_param = torch.nn.Parameter(torch.zeros([]))

    def forward(self, map, room_mask, steps_remaining):
        X = map.to(torch.float32)
        for layer in self.map_sequential:
            X = layer(X)

        X = torch.cat([X, steps_remaining.view(-1, 1), room_mask], dim=1)
        for layer in self.fc_sequential:
            X = layer(X)
        return X[:, 0]


class TrainingSession():
    def __init__(self, env: MazeBuilderEnv,
                 value_network: torch.nn.Module,
                 policy_network: PolicyNetwork,
                 value_optimizer: torch.optim.Optimizer,
                 policy_optimizer: torch.optim.Optimizer,
                 ):
        self.env = env
        self.value_network = value_network
        self.policy_network = policy_network
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer
        self.num_rounds = 0

    def generate_round(self, episode_length, render=False):
        map, room_mask = self.env.reset()
        map_list = [map]
        room_mask_list = [room_mask]
        position_list = []
        direction_list = []
        action_list = []
        reward_list = []
        self.policy_network.eval()
        for j in range(episode_length):
            if render:
                self.env.render()
            position, direction, left_ids, right_ids, down_ids, up_ids = env.choose_random_door()
            steps_remaining = torch.full_like(direction, episode_length - j)
            with torch.no_grad():
                policy_out = self.policy_network(map, room_mask, position, direction,
                                                 left_ids, right_ids, down_ids, up_ids, steps_remaining)
            left_raw_logprobs, right_raw_logprobs, down_raw_logprobs, up_raw_logprobs = policy_out
            reward, map, room_mask, action = self.env.random_step(
                position, left_ids, right_ids, down_ids, up_ids,
                left_raw_logprobs, right_raw_logprobs, down_raw_logprobs, up_raw_logprobs)
            map_list.append(map)
            room_mask_list.append(room_mask)
            position_list.append(position)
            direction_list.append(direction)
            action_list.append(action)
            reward_list.append(reward)
        map_tensor = torch.stack(map_list, dim=0)
        room_mask_tensor = torch.stack(room_mask_list, dim=0)
        position_tensor = torch.stack(position_list, dim=0)
        direction_tensor = torch.stack(direction_list, dim=0)
        action_tensor = torch.stack(action_list, dim=0)
        reward_tensor = torch.stack(reward_list, dim=0)
        return map_tensor, room_mask_tensor, position_tensor, direction_tensor, action_tensor, reward_tensor

    def train_round(self,
                    episode_length: int,
                    batch_size: int,
                    policy_variation_penalty: float = 0.0,
                    mc_weight: float = 0.0,
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

        # Flatten the data
        n = episode_length * self.env.num_envs
        map0 = map0.view(n, 3, self.env.map_x, self.env.map_y)
        map1 = map1.view(n, 3, self.env.map_x, self.env.map_y)
        room_mask0 = room_mask0.view(n, len(self.env.rooms))
        room_mask1 = room_mask1.view(n, len(self.env.rooms))
        position = position.view(n, 2)
        direction = direction.view(n)
        action = action.view(n)
        reward = reward.view(n)
        cumul_reward = cumul_reward.view(n)
        steps_remaining = steps_remaining.view(n)

        # Shuffle the data
        perm = torch.randperm(n)
        map0 = map0[perm, :, :, :]
        map1 = map1[perm, :, :, :]
        room_mask0 = room_mask0[perm, :]
        room_mask1 = room_mask1[perm, :]
        position = position[perm]
        direction = direction[perm]
        action = action[perm]
        reward = reward[perm]
        cumul_reward = cumul_reward[perm]
        steps_remaining = steps_remaining[perm]

        num_batches = n // batch_size

        # Make one pass through the data, updating the value network
        total_value_loss_bs = 0.0
        total_value_loss_mc = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            # map0_batch = map0[start:end, :, :, :]
            # room_mask0_batch = room_mask0[start:end, :]
            # cumul_reward_batch = cumul_reward[start:end]
            # steps_remaining_batch = steps_remaining[start:end]
            # value0 = self.value_network(map0_batch, room_mask0_batch, steps_remaining_batch)
            # value_loss = torch.mean((value0 - cumul_reward_batch) ** 2)
            map0_batch = map0[start:end, :, :, :]
            map1_batch = map1[start:end, :, :, :]
            room_mask0_batch = room_mask0[start:end, :]
            room_mask1_batch = room_mask1[start:end, :]
            reward_batch = reward[start:end]
            cumul_reward_batch = cumul_reward[start:end]
            steps_remaining_batch = steps_remaining[start:end]
            value0 = self.value_network(map0_batch, room_mask0_batch, steps_remaining_batch)
            with torch.no_grad():
                value1 = self.value_network(map1_batch, room_mask1_batch, steps_remaining_batch - 1)
                target = torch.where(steps_remaining_batch == 1, reward_batch.to(torch.float32), reward_batch + value1)
            value_loss_bs = torch.mean((value0 - target) ** 2)
            value_loss_mc = torch.mean((value0 - cumul_reward_batch) ** 2)
            value_loss = (1 - mc_weight) * value_loss_bs + mc_weight * value_loss_mc

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1e-5)
            self.value_optimizer.step()
            # self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
            total_value_loss_bs += value_loss_bs.item()
            total_value_loss_mc += value_loss_mc.item()

        # Make a second pass through the data, updating the policy network
        total_policy_loss = 0.0
        total_policy_variation = 0.0
        self.policy_network.train()
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            map0_batch = map0[start:end, :, :, :]
            map1_batch = map1[start:end, :, :, :]
            room_mask0_batch = room_mask0[start:end, :]
            room_mask1_batch = room_mask1[start:end, :]
            steps_remaining_batch = steps_remaining[start:end]
            reward_batch = reward[start:end]
            action_batch = action[start:end]
            position_batch = position[start:end, :]
            direction_batch = direction[start:end]
            with torch.no_grad():
                value0 = self.value_network(map0_batch, room_mask0_batch, steps_remaining_batch)
                value1 = self.value_network(map1_batch, room_mask1_batch, steps_remaining_batch - 1)
            advantage = value1 + reward_batch - value0
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
            # self.policy_network.decay(weight_decay * self.policy_optimizer.param_groups[0]['lr'])
            total_policy_loss += policy_loss.item()
            total_policy_variation += policy_variation.item()

        self.num_rounds += 1

        return mean_reward, max_reward, cnt_max_reward, total_value_loss_bs / num_batches, \
               total_value_loss_mc / num_batches, total_policy_loss / num_batches, total_policy_variation / num_batches


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

# device = torch.device('cpu')
device = torch.device('cuda:0')

num_envs = 256
# num_envs = 1
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
map_x = 32
map_y = 24
# map_x = 10
# map_y = 10
env = MazeBuilderEnv(rooms,
                     map_x=map_x,
                     map_y=map_y,
                     num_envs=num_envs,
                     device=device)
print(env.room_tensor.shape, env.left_door_tensor.shape, env.right_door_tensor.shape, env.down_door_tensor.shape, env.up_door_tensor.shape)


value_network = ValueNetwork(env.room_tensor,
                             map_x=map_x,
                             map_y=map_y,
                             map_channels=[64, 64, 128],
                             map_kernel_size=[11, 9, 5],
                             fc_widths=[128, 128, 128],
                             batch_norm_momentum=0.1,
                             ).to(device)
policy_network = PolicyNetwork(env.room_tensor, env.left_door_tensor, env.right_door_tensor,
                               env.down_door_tensor, env.up_door_tensor,
                               map_x=map_x,
                               map_y=map_y,
                               # local_radius=5,
                               map_channels=[64, 64, 128],
                               map_kernel_size=[11, 9, 5],
                               fc_widths=[128, 128, 128],
                               door_embedding_width=128,
                               batch_norm_momentum=0.1,
                               ).to(device)
policy_network.fc_sequential[-1].weight.data[:, :] = 0.0
policy_network.fc_sequential[-1].bias.data[:] = 0.0
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.0002, betas=(0.5, 0.5), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.00002, betas=(0.5, 0.5), eps=1e-15)

print(value_network)
print(value_optimizer)
print(policy_network)
print(policy_optimizer)
logging.info("Starting training")

session = TrainingSession(env,
                          value_network=value_network,
                          policy_network=policy_network,
                          value_optimizer=value_optimizer,
                          policy_optimizer=policy_optimizer)

torch.set_printoptions(linewidth=120, threshold=10000)

# map_tensor, room_mask_tensor, position_tensor, direction_tensor, action_tensor, reward_tensor = session.generate_round(
#     episode_length=episode_length,
#     render=True)

#
# # session = pickle.load(open('models/crateria-2021-06-29T13:35:06.399214.pkl', 'rb'))
#
# import io
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name =='_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)
# session = CPU_Unpickler(open('models/crateria-2021-07-09T20:58:34.290741.pkl', 'rb')).load()
# session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
batch_size = 2 ** 8
# batch_size = 2 ** 13  # 2 ** 12
policy_variation_penalty = 0.01
session.env = env
# session.value_optimizer.param_groups[0]['lr'] = 0.00005
# session.policy_optimizer.param_groups[0]['lr'] = 0.00001
# session.value_optimizer.param_groups[0]['betas'] = (0.5, 0.9)

logging.info(
    "num_envs={}, batch_size={}, policy_variation_penalty={}".format(session.env.num_envs, batch_size,
                                                                     policy_variation_penalty))
for i in range(10000):
    mean_reward, max_reward, cnt_max_reward, value_loss_bs, value_loss_mc, policy_loss, policy_variation = session.train_round(
        episode_length=episode_length,
        batch_size=batch_size,
        policy_variation_penalty=policy_variation_penalty,
        mc_weight=0.2,
        # render=True)
        render=False)
    # render=i % display_freq == 0)
    logging.info("{}: reward={:.3f} (max={:d}, cnt={:d}), value_loss={:.5f} (mc={:.5f}), policy_loss={:.5f}, policy_variation={:.5f}".format(
        session.num_rounds, mean_reward, max_reward, cnt_max_reward, value_loss_bs, value_loss_mc, policy_loss, policy_variation))
    pickle.dump(session, open(pickle_name, 'wb'))


# while True:
#     map_tensor, room_mask_tensor, position_tensor, direction_tensor, action_tensor, reward_tensor = session.generate_round(64, render=False)
#     print(torch.sum(reward_tensor), torch.sum(room_mask_tensor[-1, 0, :]))
#     if torch.sum(reward_tensor) >= 32:
#         break
# session.env.render()



# # state = env.reset()
# # for j in range(episode_length):
# #     with torch.no_grad():
# #         raw_p = session.policy_network(state)
# #     log_p = raw_p - torch.logsumexp(raw_p, dim=1, keepdim=True)
# #     p = torch.exp(log_p)
# #     cumul_p = torch.cumsum(p, dim=1)
# #     rnd = torch.rand([session.env.num_envs, 1])
# #     action = torch.clamp(torch.searchsorted(cumul_p, rnd), max=session.env.num_actions - 1)
# #     reward, state = session.env.step(action.squeeze(1))
# #     session.env.render()
#
