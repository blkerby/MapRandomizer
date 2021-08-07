import torch

class ReplayBuffer:
    def __init__(self, capacity, storage_device, retrieval_device):
        self.capacity = capacity
        self.tensor_list = None
        self.position = 0
        self.size = 0
        self.storage_device = storage_device
        self.retrieval_device = retrieval_device

    def initial_allocation(self, input_tensor_list):
        self.tensor_list = []
        for input_tensor in input_tensor_list:
            shape = list(input_tensor.shape)
            shape[0] = self.capacity
            tensor = torch.zeros(shape, dtype=input_tensor.dtype, device=self.storage_device)
            self.tensor_list.append(tensor)

    def insert(self, input_tensor_list, randomized=False):
        if self.tensor_list is None:
            self.initial_allocation(input_tensor_list)
        assert len(input_tensor_list) == len(self.tensor_list)
        size = input_tensor_list[0].shape[0]
        if randomized:
            assert self.size == self.capacity
            indices = torch.randint(high=self.size, size=[size], device=self.tensor_list[0].device)
            for i in range(len(input_tensor_list)):
                self.tensor_list[i][indices] = input_tensor_list[i].to(self.storage_device)
        else:
            remaining = self.capacity - self.position
            size_to_use = min(size, remaining)
            for i in range(len(input_tensor_list)):
                self.tensor_list[i][self.position:(self.position + size_to_use)] = input_tensor_list[i][:size_to_use]
            self.position += size_to_use
            self.size = max(self.size, self.position)
            if self.position == self.capacity:
                self.position = 0

    def sample(self, n):
        indices = torch.randint(high=self.size, size=[n], device=self.tensor_list[0].device)
        return [tensor[indices].to(self.retrieval_device) for tensor in self.tensor_list]
