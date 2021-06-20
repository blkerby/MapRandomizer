import tkinter
import tkinter.font
from typing import List, Tuple
import numpy as np
from maze_builder.types import Room
from maze_builder.crateria import rooms

LEFT_ARROW = '\u2190'
UP_ARROW = '\u2191'
RIGHT_ARROW = '\u2192'
DOWN_ARROW = '\u2193'

def make_color(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

class Map:
    def __init__(self, tile_x, tile_y):
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_width = 30
        self.margin = 50
        self.font_size = int(self.tile_width * 0.8)
        self.width = tile_x * self.tile_width + 2 * self.margin
        self.height = tile_y * self.tile_width + 2 * self.margin
        self.root = tkinter.Tk()
        self.root.title("Map randomizer")
        self.canvas = tkinter.Canvas(self.root, bg="white", width=self.width, height=self.height)
        self.canvas.pack()
        default_font = tkinter.font.nametofont("TkDefaultFont")
        default_font.configure(size=self.font_size)

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

    def display_rooms_interior(self, rooms: List[Room], xs: List[int], ys: List[int], colors):
        inverted_colors = [[[0, 0, 0] for _ in range(map.tile_x)] for _ in range(map.tile_y)]
        for k, room in enumerate(rooms):
            for i in range(room.height):
                for j in range(room.width):
                    if room.map[i][j] == 1:
                        c = inverted_colors[ys[k] + i][xs[k] + j]
                        c[0] = min(0xff, c[0] + 0xff - colors[k][0])
                        c[1] = min(0xff, c[1] + 0xff - colors[k][1])
                        c[2] = min(0xff, c[2] + 0xff - colors[k][2])
        for y in range(map.tile_y):
            for x in range(map.tile_x):
                # multiplicity_cap = max(self.interior_colors.keys())
                # multiplicity_capped = multiplicity[y][x] if multiplicity[y][x] <= multiplicity_cap else multiplicity_cap
                # color = self.interior_colors[multiplicity_capped]
                inverted_color = inverted_colors[y][x]
                print(inverted_color)
                color = make_color(0xff - inverted_color[0], 0xff - inverted_color[1], 0xff - inverted_color[2])
                x0 = self.margin + x * self.tile_width
                y0 = self.margin + y * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    def display_rooms(self, rooms: List[Room], xs: List[int], ys: List[int], colors: List[Tuple[int, int, int]]):
        self.display_rooms_interior(rooms, xs, ys, colors)
        for k, room in enumerate(rooms):
            self.display_room_borders(room, xs[k], ys[k])
            self.display_room_doors(room, xs[k], ys[k])

# map_width = 60
# map_height = 40
map_width = 30
map_height = 20
map = Map(map_width, map_height)
x = 0
y = 0
xs = []
ys = []
colors = []
for room in rooms:
    room.height = len(room.map)
    room.width = len(room.map[0])
    x = np.random.randint(map_width - room.width)
    y = np.random.randint(map_height - room.height)
    # map.display_room(room, x, y)
    xs.append(x)
    ys.append(y)
    c = np.random.randint(3)
    if c == 0:
        colors.append([0xf0, 0xb0, 0xb0])
    elif c == 1:
        colors.append([0xb0, 0xf0, 0xb0])
    elif c == 2:
        colors.append([0xb0, 0xb0, 0xf0])
    x += len(room.map[0]) + 1
    if x >= map_width - 10:
        x = 0
        y += 11
map.display_rooms(rooms, xs, ys, colors)
map.root.update()
map.root.mainloop()