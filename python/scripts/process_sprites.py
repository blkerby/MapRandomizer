import logging
import json
import argparse
import os
import ips_util
from PIL import Image

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("base_rom", help="path to base ROM")
parser.add_argument("sprites", default="", help="sprite names to process (comma-separated list)")
args = parser.parse_args()

sprite_path = "MapRandoSprites/samus_sprites"
manifest = json.load(open(f"{sprite_path}/manifest.json", "r"))
manifest = [sprite for category in manifest for sprite in category['sprites']]
if args.sprites != "":
    sprite_set = set(args.sprites.split(","))
    manifest = [[sprite for sprite in category if sprite["name"] in sprite_set] for category in manifest]

def create_static_thumbnails():
    logging.info("Creating static thumbnails")
    output_path = "rust/static/samus_sprites"
    for sprite_json in manifest:
        print(sprite_json['name'])
        sprite_sheet_filename = sprite_json['name'] + '.png'
        output_filename = sprite_sheet_filename
        output_file = f"{output_path}/{sprite_sheet_filename}"
        sheet = Image.open(f"{sprite_path}/{output_filename}")
        # Use facing-forward loading pose (Power Suit)
        if sprite_json['name'] == 'samus_cannon':
            x0, y0 = 532, 201
        else:
            x0, y0 = 405, 143
        x1 = x0 + 32
        y1 = y0 + 56
        x_padding = 4
        featured_pose = sheet.crop((x0 - x_padding, y0, x1 + x_padding, y1))
        # add padding on the sides to make the image sizes match between static and animated
        for x in list(range(x_padding)) + [x1 - x0 + x_padding + i for i in range(x_padding)]:
            for y in range(0, y1 - y0):
                featured_pose.putpixel((x, y), featured_pose.getpixel((x_padding, 0)))
        featured_pose.save(output_file)


def create_animated_thumbnails():
    logging.info("Creating animated thumbnails")
    output_path = "rust/static/samus_sprites"
    for sprite_json in manifest:
        sprite_name = sprite_json['name']
        print(sprite_name)
        sprite_sheet_filename = sprite_name + '.png'
        output_filename = sprite_name + '.gif'
        output_file = f"{output_path}/{output_filename}"
        sheet = Image.open(f"{sprite_path}/{sprite_sheet_filename}")

        # Extract walking right animation:
        if sprite_name == 'samus_cannon':
            x, y = 439, 617
        else:
            x, y = 439, 509
        y_padding = 4
        frames = []
        for i in range(10):
            x0, y0, x1, y1 = x, y, x + 40, y + 48
            frame = sheet.crop((x0, y0 - y_padding, x1, y1 + y_padding))
            # add padding on the top/bottom to make the image sizes match between static and animated
            for yp in list(range(y_padding)) + [y1 - y0 + y_padding + j for j in range(y_padding)]:
                for xp in range(0, x1 - x0):
                    frame.putpixel((xp, yp), frame.getpixel((0, y_padding)))
            frames.append(frame)
            x += 42
        t = (1000 / 60) / 0.7  # animate at 70% speed
        durations = [2*t, 3*t, 2*t, 3*t, 2*t, 3*t, 2*t, 3*t, 2*t, 3*t]
        frames[0].save(output_file, save_all=True, append_images=frames[1:], loop=0, disposal=2, duration=durations)

def create_patches():
    logging.info("Creating sprite IPS patches")
    output_path = "patches/samus_sprites/"
    os.chdir("SpriteSomething")
    for sprite_json in manifest:
        sprite_sheet_filename = sprite_json['name'] + '.png'
        output_patch_filename = sprite_json['name'] + '.ips'
        tmpfile = "/tmp/out.sfc"
        logging.info("Processing {}".format(sprite_sheet_filename))
        os.system("python SpriteSomething.py "
                  "--cli=1 --mode=inject-new "
                  f"--sprite=../{sprite_path}/{sprite_sheet_filename} "
                  f"--dest-filename={tmpfile} "
                  "--src-filepath=. "
                  f"--src-filename={args.base_rom} ")
        os.system(f"ips_util create -o ../{output_path}/{output_patch_filename} {args.base_rom} {tmpfile}")
    os.chdir("..")


create_static_thumbnails()
create_animated_thumbnails()
# create_patches()
logging.info("Done!")
