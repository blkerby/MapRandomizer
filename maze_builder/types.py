from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Room:
    name: str
    map: List[List[int]]
    door_left: Optional[List[List[int]]] = None
    door_right: Optional[List[List[int]]] = None
    door_up: Optional[List[List[int]]] = None
    door_down: Optional[List[List[int]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    area: int = 0

    def populate(self):
        self.height = len(self.map)
        self.width = len(self.map[0])
        if self.door_left is None:
            self.door_left = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_right is None:
            self.door_right = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_down is None:
            self.door_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.door_up is None:
            self.door_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
