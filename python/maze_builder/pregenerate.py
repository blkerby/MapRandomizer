import concurrent.futures

import util
import torch
import torch.profiler
import logging
from maze_builder.types import EnvConfig, EpisodeData, reconstruct_room_data
from maze_builder.env import MazeBuilderEnv
from maze_builder.model import TransformerModel
from maze_builder.train_session import TrainingSession
import logic.rooms.crateria
from datetime import datetime
import pickle
import logic.rooms.crateria_isolated
# import logic.rooms.norfair_isolated


start_time = datetime.now()
logging.basicConfig(format='%(asctime)s %(message)s',
                    # level=logging.DEBUG,
                    level=logging.INFO,
                    handlers=[logging.FileHandler("pregenerate.log"),
                              logging.FileHandler(f"logs/pregenerate-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])

devices = [torch.device('cuda:0'), torch.device('cuda:1')]
num_devices = len(devices)
device = devices[0]
executor = concurrent.futures.ThreadPoolExecutor(len(devices))

num_envs = 2 ** 11
rooms = logic.rooms.crateria_isolated.rooms
# rooms = logic.rooms.norfair_isolated.rooms
# rooms = logic.rooms.all_rooms.rooms
episode_length = len(rooms)

cpu_executor = None


map_x = 32
map_y = 32
# map_x = 48
# map_y = 48
# map_x = 64
# map_y = 64

env_config = EnvConfig(
    rooms=rooms,
    map_x=map_x,
    map_y=map_y,
)
envs = [MazeBuilderEnv(rooms,
                       map_x=map_x,
                       map_y=map_y,
                       num_envs=num_envs,
                       device=device,
                       must_areas_be_connected=False,
                       starting_room_name="Landing Site")
                       # starting_room_name="Business Center")
        for device in devices]


dummy_model = TransformerModel(
    rooms=envs[0].rooms,
    num_doors=envs[0].num_doors,
    num_outputs=envs[0].num_doors + envs[0].num_missing_connects + envs[0].num_doors + envs[0].num_non_save_dist + 1 + envs[0].num_missing_connects + 1,
    map_x=env_config.map_x,
    map_y=env_config.map_y,
    block_size_x=8,
    block_size_y=8,
    embedding_width=1,
    key_width=1,
    value_width=1,
    attn_heads=1,
    hidden_width=1,
    arity=1,
    num_local_layers=0,
    embed_dropout=0.0,
    ff_dropout=0.0,
    attn_dropout=0.0,
    num_global_layers=0,
    global_attn_heads=1,
    global_attn_key_width=1,
    global_attn_value_width=1,
    global_width=1,
    global_hidden_width=1,
    global_ff_dropout=0.0,
    use_action=True,
).to(device)


optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.00005, betas=(0.9, 0.9), eps=1e-5)
session = TrainingSession(envs,
                          state_model=dummy_model,
                          action_model=dummy_model,
                          state_optimizer=optimizer,
                          action_optimizer=optimizer,
                          data_path="data/pregen-{}".format(start_time.isoformat()),
                          ema_beta=0.999,
                          episodes_per_file=num_envs * num_devices,
                          decay_amount=0.0)

print_freq = 128

total_reward = 0.0
cnt_rounds = 0
logging.info("Generating to {}".format(session.replay_buffer.data_path))
for i in range(1000000):
    with util.DelayedKeyboardInterrupt():
        data = session.generate_round(
            episode_length=episode_length,
            num_candidates_min=1,
            num_candidates_max=1,
            temperature=torch.full([num_envs], 1.0),
            temperature_decay=1.0,
            explore_eps=0.0,
            compute_cycles=False,
            balance_coef=0.0,
            save_dist_coef=0.0,
            graph_diam_coef=0.0,
            mc_dist_coef=torch.full([num_envs], 0.0),
            toilet_good_coef=0.0,
            executor=executor,
            cpu_executor=cpu_executor,
            render=False)

        total_reward += float(torch.mean(data.reward.to(torch.float32)))
        cnt_rounds += 1

        session.replay_buffer.insert(data)
        session.num_rounds += 1

        if session.num_rounds % print_freq == 0:
            mean_reward = total_reward / cnt_rounds
            logging.info("{}: {:.3f}".format(session.num_rounds, mean_reward))