from maze_builder.display import MapDisplay
import json

map = json.load(open('../maps/v110c-tame/10004.json', 'r'))

display = MapDisplay(72, 72, 20)
display.display_vanilla_areas(map)
# display.display_assigned_areas(map)
# display.display_assigned_areas_with_saves(map)
display.image.show()
