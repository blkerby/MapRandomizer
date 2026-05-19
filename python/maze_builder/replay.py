from typing import Optional
import torch
from typing import List
from maze_builder.types import EpisodeData
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
        self.num_episodes = 0
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
        file_tensor_list = []
        for file_num in file_num_list:
            file_path = os.path.join(self.data_path, "{}.pkl".format(file_num))
            data = pickle.load(open(file_path, 'rb'))
            n = data.reward.shape[0]
            file_tensor_list.append(torch.full([n], file_num, dtype=torch.int64))
            data_list.append(data)

        file_num_tensor = torch.cat(file_tensor_list)
        out = {}
        for field in EpisodeData.__dataclass_fields__.keys():
            tensor_list = []
            for data in data_list:
                tensor_list.append(getattr(data, field))
            combined_tensor = torch.cat(tensor_list, dim=0)
            out[field] = combined_tensor
        return EpisodeData(**out), file_num_tensor

    def insert(self, episode_data: EpisodeData):
        n = episode_data.reward.shape[0]
        assert n == self.episodes_per_file
        self.add_file(episode_data)
        self.num_episodes += n

    def sample(self, batch_size, num_batches, hist_frac, hist_c, hist_max, env) -> List[EpisodeData]:
        device = env.device
        n = batch_size * num_batches
        # num_files = n // self.episodes_per_file
        num_files = max(0, n // self.episodes_per_file - 1)

        # if num_files > self.num_files:
        #     num_files = self.num_files
        # file_num_list = random.sample(list(range(self.num_files)), num_files)

        hist_frac = min(hist_frac, hist_max / (self.num_files * self.episodes_per_file))
        t = torch.pow(torch.rand([num_files]), 1 / (1 + hist_c)) * hist_frac + (1 - hist_frac)
        # file_num_list = torch.randint(int((1 - hist_frac) * self.num_files), self.num_files, [num_files]).tolist()
        file_num_list = torch.floor(t * self.num_files).to(torch.int64).clamp_max(self.num_files - 1).tolist()
        
        # Always include the most recent file, so that we train immediately on the most recent data
        file_num_list = [self.num_files - 1] + file_num_list
    
        data, _ = self.read_files(file_num_list)
        batch_list = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > data.reward.shape[0]:
                break
            batch_list.append(EpisodeData(
                action=data.action[start:end].to(device),
                map_door_id=data.map_door_id[start:end].to(device),
                room_door_id=data.room_door_id[start:end].to(device),
                door_connects=data.door_connects[start:end].to(device),
                missing_connects=data.missing_connects[start:end].to(device),
                save_distances=data.save_distances[start:end].to(device),
                graph_diameter=data.graph_diameter[start:end].to(device),
                mc_distances=data.mc_distances[start:end].to(device),
                toilet_good=data.toilet_good[start:end].to(device),
                cycle_cost=data.cycle_cost[start:end].to(device),
                reward=data.reward[start:end].to(device),
                temperature=data.temperature[start:end].to(device),
                mc_dist_coef=data.mc_dist_coef[start:end].to(device),
                prob=data.prob[start:end].to(device),
                prob0=data.prob0[start:end].to(device),
                cand_count=data.cand_count[start:end].to(device),
                test_loss=data.test_loss[start:end].to(device),
            ))
        return batch_list
