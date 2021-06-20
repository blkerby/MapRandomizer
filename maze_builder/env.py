from typing import List
import numpy as np
from maze_builder.reward import compute_reward
from maze_builder.types import Room
from maze_builder.display import MapDisplay
import gym

class MazeBuilderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rooms: List[Room], map_x: int, map_y: int, action_radius: int):
        for room in rooms:
            room.populate()

        self.rooms = rooms
        self.room_arrays = [np.stack([np.array(room.map).T,
                                      np.array(room.door_left).T,
                                      np.array(room.door_right).T,
                                      np.array(room.door_down).T,
                                      np.array(room.door_up).T])
                                     for room in rooms]
        self.map_x = map_x
        self.map_y = map_y
        self.cap_x = np.array([map_x - room.width for room in rooms])
        self.cap_y = np.array([map_y - room.height for room in rooms])
        self.cap = np.stack([self.cap_x, self.cap_y], axis=1)
        assert (self.cap > 0).all()  # Ensure map is big enough for largest room in each direction
        self.action_radius = action_radius
        self.action_width = 2 * action_radius + 1

        self.action_space = gym.spaces.Discrete(len(rooms) * self.action_width ** 2)
        self.observation_space = gym.spaces.Box(low=0, high=max(map_x, map_y), shape=[len(rooms), 2], dtype=int)
        self.state = None
        self.map_display = None
        self.color_map = {0: (0xd0, 0x90, 0x90)}

    def reset(self):
        self.state = np.random.randint(low=0, high=2 ** 30, size=[len(self.rooms), 2]) % self.cap
        return self.state

    def step(self, action: int):
        room_index = action // self.action_width ** 2
        displacement = action % self.action_width ** 2
        displacement_x = displacement % self.action_width - self.action_radius
        displacement_y = displacement // self.action_width - self.action_radius
        self.state[room_index, 0] += displacement_x
        self.state[room_index, 1] += displacement_y
        self.state[room_index, 0] = max(0, min(self.cap_x[room_index], self.state[room_index, 0]))
        self.state[room_index, 1] = max(0, min(self.cap_y[room_index], self.state[room_index, 1]))

        reward = maze_builder.reward.compute_reward(self.room_arrays, self.state, self.map_x, self.map_y)
        done = (reward == 0)
        return self.state, reward, done, {}

    def render(self, mode='human'):
        if self.map_display is None:
            self.map_display = MapDisplay(self.map_x, self.map_y)
        xs = list(self.state[:, 0])
        ys = list(self.state[:, 1])
        colors = [self.color_map[room.area] for room in self.rooms]
        self.map_display.display(self.rooms, xs, ys, colors)

    def close(self):
        pass

import maze_builder.crateria
env = MazeBuilderEnv(maze_builder.crateria.rooms, map_x=40, map_y=20, action_radius=2)
obs = env.reset()
for i in range(100000):
    env.render()
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
# env.map_display.root.mainloop()