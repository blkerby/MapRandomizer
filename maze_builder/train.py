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
pickle_name = 'models/crateria-{}.pkl'.format(start_time.isoformat())
logging.info("Checkpoint path: {}".format(pickle_name))


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=[2, 3])


class PolicyNetwork(torch.nn.Module):
    def __init__(self, room_tensor, left_door_tensor, right_door_tensor, down_door_tensor, up_door_tensor):
        super().__init__()
        self.room_tensor = room_tensor
        self.left_door_tensor = left_door_tensor
        self.right_door_tensor = right_door_tensor
        self.down_door_tensor = down_door_tensor
        self.up_door_tensor = up_door_tensor
        self.dummy_param = torch.nn.Parameter(torch.zeros([]))

    def forward(self, map, room_mask, position, left_ids, right_ids, down_ids, up_ids, steps_remaining):
        device = map.device
        left_raw_logprobs = torch.zeros([left_ids.shape[0], self.right_door_tensor.shape[0]], dtype=torch.float32,
                                        device=device)
        right_raw_logprobs = torch.zeros([right_ids.shape[0], self.left_door_tensor.shape[0]], dtype=torch.float32,
                                         device=device)
        down_raw_logprobs = torch.zeros([down_ids.shape[0], self.up_door_tensor.shape[0]], dtype=torch.float32,
                                        device=device)
        up_raw_logprobs = torch.zeros([up_ids.shape[0], self.down_door_tensor.shape[0]], dtype=torch.float32,
                                      device=device)
        return left_raw_logprobs, right_raw_logprobs, down_raw_logprobs, up_raw_logprobs


class ValueNetwork(torch.nn.Module):
    def __init__(self, room_tensor, map_channels, map_kernel_size, fc_widths):
        super().__init__()
        self.room_tensor = room_tensor

        map_layers = []
        map_channels = [3] + map_channels
        for i in range(len(map_channels) - 1):
            map_layers.append(torch.nn.Conv2d(map_channels[i], map_channels[i + 1],
                                              kernel_size=(map_kernel_size[i], map_kernel_size[i]),
                                              padding=map_kernel_size[i] // 2))
            map_layers.append(torch.nn.ReLU())
            # map_layers.append(torch.nn.MaxPool2d(3, stride=2))
        map_layers.append(GlobalAvgPool2d())
        self.map_sequential = torch.nn.Sequential(*map_layers)

        fc_layers = []
        fc_widths = [map_channels[-1] + 1 + room_tensor.shape[0]] + fc_widths
        for i in range(len(fc_widths) - 1):
            fc_layers.append(torch.nn.Linear(fc_widths[i], fc_widths[i + 1]))
            fc_layers.append(torch.nn.ReLU())
        fc_layers.append(torch.nn.Linear(fc_widths[-1], 1))
        self.fc_sequential = torch.nn.Sequential(*fc_layers)
        # self.lin = torch.nn.Linear(1, 1)
        # self.dummy_param = torch.nn.Parameter(torch.zeros([]))

    def forward(self, map, room_mask, steps_remaining):
        X = map.to(torch.float32)
        for layer in self.map_sequential:
            # print(X.shape, layer)
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

    def generate_round(self, episode_length, render=False):
        map, room_mask = self.env.reset()
        map_list = [map]
        room_mask_list = [room_mask]
        position_list = []
        direction_list = []
        action_list = []
        reward_list = []
        for j in range(episode_length):
            if render:
                self.env.render()
            position, direction, left_ids, right_ids, down_ids, up_ids = env.choose_random_door()
            steps_remaining = episode_length - j
            with torch.no_grad():
                policy_out = self.policy_network(map, room_mask, position, left_ids, right_ids, down_ids, up_ids,
                                                 steps_remaining)
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
        steps_remaining = (episode_length - torch.arange(episode_length, device=map.device)).view(-1, 1).repeat(1, env.num_envs)

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
        total_value_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            map0_batch = map0[start:end, :, :, :]
            room_mask0_batch = room_mask0[start:end, :]
            cumul_reward_batch = cumul_reward[start:end]
            steps_remaining_batch = steps_remaining[start:end]
            value0 = self.value_network(map0_batch, room_mask0_batch, steps_remaining_batch)
            value_loss = torch.mean((value0 - cumul_reward_batch) ** 2)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1e-5)
            self.value_optimizer.step()
            # self.value_network.decay(weight_decay * self.value_optimizer.param_groups[0]['lr'])
            total_value_loss += value_loss.item()

        # # Make a second pass through the data, updating the policy network
        # total_policy_loss = 0.0
        # total_policy_variation = 0.0
        # for i in range(num_batches):
        #     start = i * batch_size
        #     end = (i + 1) * batch_size
        #     state0_batch = state0[start:end, :, :]
        #     state1_batch = state1[start:end, :, :]
        #     action_batch = action[start:end, :]
        #     with torch.no_grad():
        #         value0 = decode_map(self.value_network(state0_batch), state0_batch, self.env.room_tensors, aggregate=True, filter_by_channel=True)
        #         value1 = decode_map(self.value_network(state1_batch), state1_batch, self.env.room_tensors, aggregate=True, filter_by_channel=True)
        #     advantage = torch.sum(value1 - value0, dim=1)
        #     raw_p = self.policy_network(state0_batch)
        #     log_p = raw_p - torch.logsumexp(raw_p, dim=2, keepdim=True)
        #     log_p_action = log_p[
        #         torch.arange(batch_size).view(-1, 1), torch.arange(len(self.env.rooms)).view(1, -1), action_batch]
        #     policy_loss = -torch.mean(advantage * log_p_action)
        #     policy_variation = torch.mean(raw_p ** 2)
        #     policy_variation_loss = policy_variation_penalty * policy_variation
        #     self.policy_optimizer.zero_grad()
        #     (policy_loss + policy_variation_loss).backward()
        #     torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1e-5)
        #     self.policy_optimizer.step()
        #     # self.policy_network.decay(weight_decay * self.policy_optimizer.param_groups[0]['lr'])
        #     total_policy_loss += policy_loss.item()
        #     total_policy_variation += policy_variation.item()

        mean_reward = torch.sum(reward) / self.env.num_envs
        total_policy_loss = 0.0
        total_policy_variation = 0.0
        return mean_reward, total_value_loss / num_batches, total_policy_loss / num_batches, \
               total_policy_variation / num_batches


import logic.rooms.crateria

# device = torch.device('cpu')
device = torch.device('cuda:0')

num_envs = 512
# num_envs = 1024
rooms = logic.rooms.crateria.rooms
action_radius = 1
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

value_network = ValueNetwork(env.room_tensor,
                             map_channels=[32, 32, 32],
                             map_kernel_size=[9, 9, 9],
                             fc_widths=[128, 128],
                             ).to(device)
policy_network = PolicyNetwork(env.room_tensor, env.left_door_tensor, env.right_door_tensor,
                               env.down_door_tensor, env.up_door_tensor).to(device)
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.005, betas=(0.5, 0.5), eps=1e-15)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-5, betas=(0.9, 0.9), eps=1e-15)

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
# # import io
# #
# #
# # class CPU_Unpickler(pickle.Unpickler):
# #     def find_class(self, module, name):
# #         if module == 'torch.storage' and name == '_load_from_bytes':
# #             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
# #         else:
# #             return super().find_class(module, name)
# # session = CPU_Unpickler(open('models/crateria-2021-06-29T12:30:22.754523.pkl', 'rb')).load()
# session.policy_optimizer.param_groups[0]['lr'] = 5e-6
# # session.value_optimizer.param_groups[0]['betas'] = (0.8, 0.999)
batch_size = 2 ** 8
# batch_size = 2 ** 13  # 2 ** 12
policy_variation_penalty = 0.005
# session.env = env
# session.value_optimizer.param_groups[0]['lr'] = 0.005
# session.value_optimizer.param_groups[0]['betas'] = (0.5, 0.9)
logging.info(
    "num_envs={}, batch_size={}, policy_variation_penalty={}".format(session.env.num_envs, batch_size,
                                                                     policy_variation_penalty))
for i in range(10000):
    reward, value_loss, policy_loss, policy_variation = session.train_round(
        episode_length=episode_length,
        batch_size=batch_size,
        policy_variation_penalty=policy_variation_penalty,
        render=False)
    # render=i % display_freq == 0)
    logging.info("{}: reward={:.3f} value_loss={:.5f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
        i, reward, value_loss, policy_loss, policy_variation))
    # pickle.dump(session, open(pickle_name, 'wb'))
#
# # session.policy_optimizer.param_groups[0]['lr'] = 2e-5
# # horizon = 8
# # batch_size = 256
# # policy_variation_penalty = 5e-4
# # print("num_envs={}, batch_size={}, horizon={}, policy_variation_penalty={}".format(env.num_envs, batch_size, horizon, policy_variation_penalty))
# # for i in range(10000):
# #     reward, value_loss, policy_loss, policy_variation = session.train_round(
# #         num_episodes=1,
# #         episode_length=episode_length,
# #         horizon=horizon,
# #         batch_size=batch_size,
# #         weight_decay=0.0,
# #         policy_variation_penalty=policy_variation_penalty,
# #         render=False)
# #         # render=i % display_freq == 0)
# #     logging.info("{}: reward={:.3f}, value_loss={:.5f}, policy_loss={:.5f}, policy_variation={:.5f}".format(
# #         i, reward, value_loss, policy_loss, policy_variation))
#
#
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
# #
# # session.env.render()
# # out, room_infos = value_network.encode_map(env.state)
# # r = value_network.decode_map(out, room_infos)
#
# # session.env.render()
# # b = value_network._compute_room_boundaries(env.room_tensors[1][0, :, :])
# # print(env.room_tensors[1][0,:, :].t())
# # print(b[3, :, :].t())
#
# # torch.save(policy_network, "crateria_policy.pt")
# # torch.save(value_network, "crateria_value.pt")
