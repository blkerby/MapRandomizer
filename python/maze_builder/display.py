import PIL
import PIL.Image
from PIL import ImageDraw, ImageFont
from typing import List, Tuple
from maze_builder.types import Room
import logic.rooms.all_rooms

LEFT_ARROW = '\u2190'
UP_ARROW = '\u2191'
RIGHT_ARROW = '\u2192'
DOWN_ARROW = '\u2193'
LEFT_DOUBLE_ARROW = '⇐'
RIGHT_DOUBLE_ARROW = '⇒'
UP_DOUBLE_ARROW = '⇑'
DOWN_DOUBLE_ARROW = '⇓'
UP_WHITE_ARROW = '⇧'
DOWN_WHITE_ARROW = '⇩'
UP_DOUBLE_HEAD_ARROW = '↟'
DOWN_DOUBLE_HEAD_ARROW = '↡'


# def make_color(r, g, b):
#     return '#{:02x}{:02x}{:02x}'.format(r, g, b)
#
class MapDisplay:
    def __init__(self, tile_x, tile_y, tile_width):
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_width = tile_width
        self.margin = self.tile_width
        self.font_size = int(self.tile_width * 2.0)
        self.width = tile_x * self.tile_width + 2 * self.margin
        self.height = tile_y * self.tile_width + 2 * self.margin
        self.image = PIL.Image.new("RGB", (self.width, self.height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        # self.font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", self.font_size)
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', self.font_size)
        # self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', self.font_size)
        # self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', self.font_size)

        # self.font = ImageFont.truetype("Pillow/Tests/fonts/NotoSans-Regular.ttf", 48)


    def _display_room_borders(self, room: Room, x: int, y: int):
        color = (0, 0, 0)
        for i in range(room.height):
            for j in range(room.width):
                x0 = self.margin + (x + j) * self.tile_width
                y0 = self.margin + (y + i) * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                if room.map[i][j] == 1:
                    if i == 0 or room.map[i - 1][j] == 0:
                        self.draw.line((x0, y0, x1, y0), fill=color)
                    if j == 0 or room.map[i][j - 1] == 0:
                        self.draw.line((x0, y0, x0, y1), fill=color)
                    if i == room.height - 1 or room.map[i + 1][j] == 0:
                        self.draw.line((x0, y1, x1, y1), fill=color)
                    if j == room.width - 1 or room.map[i][j + 1] == 0:
                        self.draw.line((x1, y0, x1, y1), fill=color)

    def _display_room_doors(self, room: Room, x: int, y: int):
        color = (0, 0, 0)
        for i in range(room.height):
            for j in range(room.width):
                x0 = self.margin + (x + j + 0.5) * self.tile_width
                y0 = self.margin + (y + i + 0.5) * self.tile_width
                if room.door_left[i][j] == 1:
                    self.draw.text((x0, y0), text=LEFT_ARROW, fill=color, font=self.font, anchor="mm")
                if room.door_right[i][j] == 1:
                    self.draw.text((x0, y0), text=RIGHT_ARROW, fill=color, font=self.font, anchor="mm")
                if room.door_up[i][j] == 1:
                    self.draw.text((x0, y0), text=UP_ARROW, fill=color, font=self.font, anchor="mm")
                if room.door_down[i][j] == 1:
                    self.draw.text((x0, y0), text=DOWN_ARROW, fill=color, font=self.font, anchor="mm")
                if room.elevator_up[i][j] == 1:
                    self.draw.text((x0, y0), text=UP_WHITE_ARROW, fill=color, font=self.font, anchor="mm")
                if room.elevator_down[i][j] == 1:
                    self.draw.text((x0, y0), text=DOWN_WHITE_ARROW, fill=color, font=self.font, anchor="mm")
                if room.sand_up[i][j] == 1:
                    self.draw.text((x0, y0), text=UP_DOUBLE_HEAD_ARROW, fill=color, font=self.font, anchor="mm")
                if room.sand_down[i][j] == 1:
                    self.draw.text((x0, y0), text=DOWN_DOUBLE_HEAD_ARROW, fill=color, font=self.font, anchor="mm")
#
    def _display_rooms_interior(self, rooms: List[Room], xs: List[int], ys: List[int], colors):
        inverted_colors = [[[0, 0, 0] for _ in range(self.tile_x)] for _ in range(self.tile_y)]
        for k, room in enumerate(rooms):
            for i in range(room.height):
                for j in range(room.width):
                    if room.map[i][j] == 1:
                        c = inverted_colors[ys[k] + i][xs[k] + j]
                        c[0] = min(0xff, c[0] + 0xff - colors[k][0])
                        c[1] = min(0xff, c[1] + 0xff - colors[k][1])
                        c[2] = min(0xff, c[2] + 0xff - colors[k][2])
        for y in range(self.tile_y):
            for x in range(self.tile_x):
                inverted_color = inverted_colors[y][x]
                if inverted_color == [0, 0, 0]:
                    continue
                color = (0xff - inverted_color[0], 0xff - inverted_color[1], 0xff - inverted_color[2])
                x0 = self.margin + x * self.tile_width
                y0 = self.margin + y * self.tile_width
                x1 = x0 + self.tile_width
                y1 = y0 + self.tile_width
                self.draw.rectangle((x0, y0, x1, y1), fill=color, outline=color)

    def _display_grid(self):
        for i in range(self.tile_x + 1):
            x = self.margin + i * self.tile_width
            y1 = self.margin + self.tile_y * self.tile_width
            self.draw.line((x, self.margin, x, y1), fill=(0xC0, 0xC0, 0xC0))

        for i in range(self.tile_y + 1):
            y = self.margin + i * self.tile_width
            x1 = self.margin + self.tile_x * self.tile_width
            self.draw.line((self.margin, y, x1, y), fill=(0xC0, 0xC0, 0xC0))

    def display(self, rooms: List[Room], xs: List[int], ys: List[int], colors: List[Tuple[int, int, int]]):
        self._display_grid()
        self._display_rooms_interior(rooms, xs, ys, colors)
        for k, room in enumerate(rooms):
            self._display_room_borders(room, xs[k], ys[k])
            self._display_room_doors(room, xs[k], ys[k])

# map_width = 72
# map_height = 72
# map = MapDisplay(map_width, map_height, 30)
#
# rooms = logic.rooms.all_rooms.rooms[:150]
# x = 0
# y = 0
# xs = []
# ys = []
# colors = []
# for room in rooms:
#     room.populate()
#     xs.append(x)
#     ys.append(y)
#     colors.append([0xf0, 0xb0, 0xb0])
#     x += len(room.map[0]) + 1
#     if x >= map_width - 12:
#         x = 0
#         y += 5
#
# map.display(rooms, xs, ys, colors)

# map.image.show()
# map.image.save("test.png", "png")
# map.root.update()
# map.root.mainloop()