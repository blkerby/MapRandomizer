from dataclasses import dataclass
from typing import List, Optional

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