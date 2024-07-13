from typing import Optional
import torch
from typing import List
from maze_builder.types import EpisodeData, TrainingData, reconstruct_room_data
import os
import pickle
import random


class ReplayBuffer:
    def __init__(self, num_rooms, storage_device, data_path, episodes_per_file):
        self.num_rooms = num_rooms
        self.storage_device = storage_device
        self.data_path = data_path
        self.episodes_per_file = episodes_per_file
        self.num_files = 0
        os.makedirs(data_path, exist_ok=True)

    def resize(self, new_capacity: int):
        new_size = min(new_capacity, self.size)
        for field in EpisodeData.__dataclass_fields__.keys():
            current_tensor = getattr(self.episode_data, field)
            shape = list(current_tensor.shape)
            shape[0] = new_capacity
            current_tensor1 = torch.cat([current_tensor[self.position:self.size], current_tensor[:self.position]], dim=0)
            new_tensor = torch.zeros(shape, dtype=current_tensor.dtype, device=self.storage_device)
            new_tensor[:new_size] = current_tensor1[-new_size:]
            # new_tensor[:new_size] = current_tensor[:new_size]
            setattr(self.episode_data, field, new_tensor)
        self.size = new_size
        self.capacity = new_capacity
        self.position = new_size

    def add_file(self, episode_data: EpisodeData):
        next_file_number = self.num_files
        file_path = os.path.join(self.data_path, "{}.pkl".format(next_file_number))
        pickle.dump(episode_data, open(file_path, 'wb'))
        self.num_files += 1

    def read_files(self, file_num_list):
        data_list = []        
        for file_num in file_num_list:
            file_path = os.path.join(self.data_path, "{}.pkl".format(file_num))
            data = pickle.load(open(file_path, 'rb'))
            data_list.append(data)
            
        out = {}
        for field in EpisodeData.__dataclass_fields__.keys():
            tensor_list = []
            for data in data_list:
                tensor_list.append(getattr(data, field))
            combined_tensor = torch.cat(tensor_list, dim=0)
            out[field] = combined_tensor
        return EpisodeData(**out)

    def insert(self, episode_data: EpisodeData):
        n = episode_data.reward.shape[0]
        assert n == self.episodes_per_file
        self.add_file(episode_data)

    def sample(self, batch_size, num_batches, hist_frac, device: torch.device, include_next_step: bool) -> List[TrainingData]:
        n = batch_size * num_batches
        num_files = n // self.episodes_per_file

        # if num_files > self.num_files:
        #     num_files = self.num_files
        # file_num_list = random.sample(list(range(self.num_files)), num_files)
        file_num_list = torch.randint(int((1 - hist_frac) * self.num_files), self.num_files, [num_files]).tolist()

        data = self.read_files(file_num_list)
        episode_length = data.action.shape[1]
        batch_list = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > data.reward.shape[0]:
                break
            reward = data.reward[start:end]
            temperature = data.temperature[start:end]
            mc_dist_coef = data.mc_dist_coef[start:end]
            door_connects = data.door_connects[start:end, :]
            door_balance = data.door_balance[start:end, :]
            missing_connects = data.missing_connects[start:end, :]
            save_distances = data.save_distances[start:end, :]
            graph_diameter = data.graph_diameter[start:end]
            mc_distances = data.mc_distances[start:end, :]
            toilet_good = data.toilet_good[start:end]
            cycle_cost = data.cycle_cost[start:end]
            action = data.action[start:end, :, :].to(torch.int64)

            def make_batch(s):
                clamp_s = torch.clamp_max(s, data.map_door_id.shape[1] - 1)
                map_door_id = torch.where(
                    step_indices == data.map_door_id.shape[1] - 1,
                    torch.full([batch_size], -1),
                    data.map_door_id[torch.arange(start, end), clamp_s].to(torch.int64)
                )
                room_door_id = torch.where(
                    step_indices == data.room_door_id.shape[1] - 1,
                    torch.full([batch_size], -1),
                    data.room_door_id[torch.arange(start, end), clamp_s].to(torch.int64)
                )

                steps_remaining = episode_length - s
                room_mask, room_position_x, room_position_y = reconstruct_room_data(action, s, self.num_rooms)

                batch = TrainingData(
                    reward=reward.to(device),
                    door_connects=door_connects.to(device),
                    door_balance=door_balance.to(device),
                    missing_connects=missing_connects.to(device),
                    save_distances=save_distances.to(device),
                    graph_diameter=graph_diameter.to(device),
                    mc_distances=mc_distances.to(device),
                    toilet_good=toilet_good.to(device),
                    cycle_cost=cycle_cost.to(device),
                    steps_remaining=steps_remaining.to(device),
                    round_frac=torch.zeros_like(graph_diameter).to(device),
                    temperature=temperature.to(device),
                    mc_dist_coef=mc_dist_coef.to(device),
                    room_mask=room_mask.to(device),
                    room_position_x=room_position_x.to(device),
                    room_position_y=room_position_y.to(device),
                    map_door_id=map_door_id.to(device),
                    room_door_id=room_door_id.to(device),
                )
                return batch

            step_indices = torch.randint(high=episode_length, size=[batch_size])
            batch = make_batch(step_indices)

            if include_next_step:
                batch_next = make_batch(step_indices + 1)
                batch_list.append((batch, batch_next))
            else:
                batch_list.append(batch)
        return batch_list
