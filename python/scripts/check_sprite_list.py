import json
import requests

map_rando_manifest = json.load(open(f"MapRandoSprites/samus_sprites/manifest.json", "r"))
map_rando_sprite_names = set("{}.{}".format(x['name'], x['version']) for c in map_rando_manifest for x in c['sprites'])

upstream_json_str = requests.get("http://smalttpr.mymm1.com/sprites/").text
upstream_json = json.loads(upstream_json_str)
upstream_sprite_names = set(k for k, v in upstream_json['m3']['approved'].items()
                            if v['usage'] is not None and ('global' in v['usage'] or ('maprando' in v['usage'])))

upstream_sprite_names.remove("001.samus.1")
upstream_sprite_names.remove("hitboxhelper.1")  # ignoring this one, since there are two versions and we're using the other one
upstream_sprite_names.remove('super_controid.1')  # ignoring this one, since there are two versions and we're using the PG one
map_rando_sprite_names.remove("samus.1")

# Show differences between sprites in Map Rando and those in upstream listing:
print(sorted(map_rando_sprite_names.difference(upstream_sprite_names)))
print(sorted(upstream_sprite_names.difference(map_rando_sprite_names)))


# # Initial extraction
# out = []
# output_dir = "MapRandoSprites/samus_sprites/"
# for k in sorted(upstream_json['m3']['approved'].keys()):
#     print(k)
#     v = upstream_json['m3']['approved'][k]
#     if v['usage'] is None or not ('global' in v['usage'] or ('maprando' in v['usage'])):
#         continue
#     out.append({
#         'name': v['short_slug'],
#         'version': v['version'],
#         'display_name': v['name'],
#         'author': v['author'],
#     })
#     url = v['file']
#     spritesheet = requests.get(url).content
#     file = open(f"{output_dir}/{v['short_slug']}.png", 'wb')
#     file.write(spritesheet)
#     file.close()
# print(json.dumps(out))

# upstream_sprite_names = set(k for k, v in upstream_json['m3']['approved'].items() if 'usage' not in v)
