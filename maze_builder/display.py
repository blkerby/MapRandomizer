import tkinter
import tkinter.font
from typing import List
import numpy as np
from maze_builder.types import Room
from maze_builder.crateria import rooms

LEFT_ARROW = '\u2190'
UP_ARROW = '\u2191'
RIGHT_ARROW = '\u2192'
DOWN_ARROW = '\u2193'

class Map:
    def __init__(self, tile_x, tile_y):
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_width = 30
        self.margin = 50
        self.interior_colors = {
            0: '#fff',
            1: '#f88',
            2: '#c44',
            3: '#800',
        }
        self.font_size = int(self.tile_width * 0.8)
        self.width = tile_x * self.tile_width + 2 * self.margin
        self.height = tile_y * self.tile_width + 2 * self.margin
        self.root = tkinter.Tk()
        self.root.title("Map randomizer")
        self.canvas = tkinter.Canvas(self.root, bg="white", width=self.width, height=self.height)
        self.canvas.pack()
        default_font = tkinter.font.nametofont("TkDefaultFont")
        default_font.configure(size=self.font_size)

    def display_room_interior(self, room: Room, x: int, y: int):
        for i in range(room.height):
            for j in range(room.width):
                x0 = self.margin + (x + j) * self.tile_width
                y0 = self.margin + (y + i) * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                if room.map[i][j] == 1:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='#f88', outline="#f88")

    def display_room_borders(self, room: Room, x: int, y: int):
        for i in range(room.height):
            for j in range(room.width):
                x0 = self.margin + (x + j) * self.tile_width
                y0 = self.margin + (y + i) * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                if room.map[i][j] == 1:
                    if i == 0 or room.map[i - 1][j] == 0:
                        self.canvas.create_line(x0, y0, x1, y0, fill='#000')
                    if j == 0 or room.map[i][j - 1] == 0:
                        self.canvas.create_line(x0, y0, x0, y1, fill='#000')
                    if i == room.height - 1 or room.map[i + 1][j] == 0:
                        self.canvas.create_line(x0, y1, x1, y1, fill='#000')
                    if j == room.width - 1 or room.map[i][j + 1] == 0:
                        self.canvas.create_line(x1, y0, x1, y1, fill='#000')

    def display_room_doors(self, room: Room, x: int, y: int):
        for i in range(room.height):
            for j in range(room.width):
                x0 = self.margin + (x + j + 0.5) * self.tile_width
                y0 = self.margin + (y + i + 0.5) * self.tile_width
                if room.door_left is not None and room.door_left[i][j] == 1:
                    self.canvas.create_text(x0, y0, text=LEFT_ARROW)
                if room.door_right is not None and room.door_right[i][j] == 1:
                    self.canvas.create_text(x0, y0, text=RIGHT_ARROW)
                if room.door_up is not None and room.door_up[i][j] == 1:
                    self.canvas.create_text(x0, y0, text=UP_ARROW)
                if room.door_down is not None and room.door_down[i][j] == 1:
                    self.canvas.create_text(x0, y0, text=DOWN_ARROW)

    # def display_room(self, room: Room, x: int, y: int):
    #     self.display_room_interior(room, x, y)
    #     self.display_room_borders(room, x, y)
    #     self.display_room_doors(room, x, y)
    #

    def display_rooms_interior(self, rooms: List[Room], xs: List[int], ys: List[int]):
        multiplicity = [[0 for _ in range(map.tile_x)] for _ in range(map.tile_y)]
        for k, room in enumerate(rooms):
            for i in range(room.height):
                for j in range(room.width):
                    if room.map[i][j] == 1:
                        multiplicity[ys[k] + i][xs[k] + j] += 1
        for y in range(map.tile_y):
            for x in range(map.tile_x):
                multiplicity_cap = max(self.interior_colors.keys())
                multiplicity_capped = multiplicity[y][x] if multiplicity[y][x] <= multiplicity_cap else multiplicity_cap
                color = self.interior_colors[multiplicity_capped]
                x0 = self.margin + x * self.tile_width
                y0 = self.margin + y * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    def display_rooms(self, rooms: List[Room], xs: List[int], ys: List[int]):
        self.display_rooms_interior(rooms, xs, ys)
        for k, room in enumerate(rooms):
            self.display_room_borders(room, xs[k], ys[k])
            self.display_room_doors(room, xs[k], ys[k])

map_width = 60
map_height = 40
map = Map(map_width, map_height)
x = 0
y = 0
xs = []
ys = []
for room in rooms:
    room.height = len(room.map)
    room.width = len(room.map[0])
    # x = np.random.randint(map_width - room.width)
    # y = np.random.randint(map_height - room.height)
    # map.display_room(room, x, y)
    xs.append(x)
    ys.append(y)
    x += len(room.map[0]) + 1
    if x >= map_width - 10:
        x = 0
        y += 11
map.display_rooms(rooms, xs, ys)
map.root.update()
map.root.mainloop()