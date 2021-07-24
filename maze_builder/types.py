from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from logic.areas import Area, SubArea

@dataclass
class Room:
    name: str
    map: List[List[int]]
    door_left: Optional[List[List[int]]] = None
    door_right: Optional[List[List[int]]] = None
    door_down: Optional[List[List[int]]] = None
    door_up: Optional[List[List[int]]] = None
    external_door_left: Optional[List[List[int]]] = None
    external_door_right: Optional[List[List[int]]] = None
    external_door_down: Optional[List[List[int]]] = None
    external_door_up: Optional[List[List[int]]] = None
    elevator_down: Optional[List[List[int]]] = None
    elevator_up: Optional[List[List[int]]] = None
    sand_down: Optional[List[List[int]]] = None
    sand_up: Optional[List[List[int]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    area: Optional[Area] = None
    sub_area: Optional[SubArea] = None

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
        if self.external_door_left is None:
            self.external_door_left = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_right is None:
            self.external_door_right = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_down is None:
            self.external_door_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.external_door_up is None:
            self.external_door_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.elevator_down is None:
            self.elevator_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.elevator_up is None:
            self.elevator_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.sand_down is None:
            self.sand_down = [[0 for _ in range(self.width)] for _ in range(self.height)]
        if self.sand_up is None:
            self.sand_up = [[0 for _ in range(self.width)] for _ in range(self.height)]
