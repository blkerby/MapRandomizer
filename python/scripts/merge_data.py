import glob
import torch
import pickle
import os

file_num = 0
for path in sorted(glob.glob("data/2024-07-0[2-5]*/*.pkl")):
    stat = os.stat(path)
    if stat.st_size == 20702667:
        # data = pickle.load(open(path, 'rb'))
        # r = torch.mean(data.reward.to(torch.float32))
        print(path)
        os.symlink("../../" + path, f"data/pretraining/{file_num}.pkl")
        file_num += 1
