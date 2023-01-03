# TODO: Clean up this whole thing (it's a mess right now). Split stuff up into modules in some reasonable way.
import flask
from typing import List
from io import BytesIO
import numpy as np
import random
import graph_tool
import graph_tool.inference
import graph_tool.topology
from collections import defaultdict
import zipfile
import pprint
import datetime

from rando.sm_json_data import SMJsonData, GameState, Link, DifficultyConfig
from rando.items import Randomizer
from rando.escape import update_escape_timer
from logic.rooms.all_rooms import rooms
from maze_builder.types import Room, SubArea
from maze_builder.display import MapDisplay
import json
import ips_util
from rando.rom import Rom, RomRoom, area_map_ptrs, snes2pc, pc2snes, get_area_explored_bit_ptr
from rando.compress import compress
from rando.make_title_bg import encode_graphics
from rando.make_title import add_title
from rando.map_patch import MapPatcher
from rando.balance_utilities import balance_utilities
from rando.music_patch import patch_music, rerank_areas
import argparse

parser = argparse.ArgumentParser(description='Start the Map Rando web service.')
parser.add_argument('--debug', type=bool, default=False, help='Run in debug mode')
args = parser.parse_args()

VERSION = 21

import logging
from maze_builder.types import reconstruct_room_data, Direction, DoorSubtype
import logic.rooms.all_rooms
import pickle
from rando.items import ItemPlacementStrategy

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("train.log"),
                              logging.StreamHandler()])

logging.info("Debug mode: {}".format(args.debug))

import io
import os

from flask import Flask

app = Flask(__name__)

sm_json_data_path = "sm-json-data/"
sm_json_data = SMJsonData(sm_json_data_path)
map_dir = 'maps/session-2022-06-03T17:19:29.727911.pkl-bk30-subarea'
file_list = sorted(os.listdir(map_dir))



presets = [
    ('Easy', {
        'shinesparkTiles': 32,
        'resourceMultiplier': 3.0,
        'escapeTimerMultiplier': 5.0,
        'tech': [],
    }),
    ('Medium', {
        'shinesparkTiles': 28,
        'resourceMultiplier': 2.0,
        'escapeTimerMultiplier': 3.0,
        'tech': [
            'canIBJ',
            'canWalljump',
            'canShinespark',
            'canCrouchJump',
            'canDownGrab',
            'canHeatRun',
            'canSuitlessMaridia']
    }),
    ('Hard', {
        'shinesparkTiles': 24,
        'resourceMultiplier': 1.5,
        'escapeTimerMultiplier': 2.0,
        'tech': [
            'canJumpIntoIBJ',
            'canBombAboveIBJ',
            'canManipulateHitbox',
            'canUseEnemies',
            'canDamageBoost',
            'canGateGlitch',
            'canGravityJump',
            'canMockball',
            'canMidAirMockball',
            'canSpringBallJump',
            'canUseFrozenEnemies',
            'canMochtroidClimb',
            'canStationarySpinJump',
            'canMoonfall',
            'canMochtroidClip',
            'canCeilingClip',
            'canIframeSpikeJump',
            'canSingleHBJ',
            'canSnailClimb',
            'canXRayStandUp',
            'canCrumbleJump',
            'canBlueSpaceJump']
    }),
    ('Very Hard', {
        'shinesparkTiles': 20,
        'resourceMultiplier': 1.2,
        'escapeTimerMultiplier': 1.5,
        'tech': [
            'canTrickyJump',
            'canSuitlessLavaDive',
            'canSuitlessLavaWalljump',
            'canPreciseWalljump',
            'canHitbox',
            'canPlasmaHitbox',
            'canUnmorphBombBoost',
            'canLavaGravityJump',
            'can3HighMidAirMorph',
            'canTurnaroundAimCancel',
            'canStationaryMidAirMockball',
            'canTrickyUseFrozenEnemies',
            'canCrabClimb',
            'canMetroidAvoid',
            'canSandMochtroidClimb',
            'canShotBlockOverload',
            'canMaridiaTubeClip',
            'canQuickLowTideWalljumpWaterEscape',
            'canGrappleJump',
            'canDoubleHBJ',
            'canSnailClip',
            'canBombJumpBreakFree',
            'canSuperReachAround',
            'canWrapAroundShot',
            'canTunnelCrawl',
            'canSpringBallJumpMidAir',
            'canCrumbleSpinJump',
            'canIceZebetitesSkip']}),
    ('Expert', {
        'shinesparkTiles': 16,
        'resourceMultiplier': 1.0,
        'escapeTimerMultiplier': 1.2,
        'tech':
         [
             'canTrickyDashJump',
             'canCWJ',
             'canDelayedWalljump',
             'canIframeSpikeWalljump',
             'canFlatleyTurnaroundJump',
             'canContinuousDboost',
             'canReverseGateGlitch',
             'canGravityWalljump',
             'can2HighWallMidAirMorph',
             'canPixelPerfectIceClip',
             'canBabyMetroidAvoid',
             'canSunkenDualWallClimb',
             'canBreakFree',
             'canHerdBabyTurtles',
             'canSandIBJ',
             'canFastWalljumpClimb',
             'canDraygonGrappleJump',
             'canManipulateMellas',
             'canMorphlessTunnelCrawl',
             'canSpringwall',
             'canDoubleSpringBallJumpMidAir',
             'canXRayClimb',
             'canLeftFacingDoorXRayClimb',
             'canQuickCrumbleEscape',
             'canSpeedZebetitesSkip',
             'canRemorphZebetiteSkip',
             'canBePatient',
             'canInsaneWalljump',
             'canNonTrivialIceClip',
             'canBeetomClip',
             'canWallCrawlerClip',
             'canPuyoClip',
             'canMultiviolaClip',
             'canRightFacingDoorXRayClimb']
     })
]

# Tech which is currently not used by any strat in logic, so we avoid showing on the website:
ignored_tech_set = {
    'canSpaceTime',
    'canGTCode',
    'canXRayClimbOOB',
    'canWallIceClip',
    'canGrappleClip',
    'canUseSpeedEchoes',
    'canCrystalFlash',
    'canCrystalFlashForceStandup'
}

item_placement_strategies = {
    'Open': ItemPlacementStrategy.OPEN,
    'Closed': ItemPlacementStrategy.CLOSED,
}

def get_tech_description(name):
    desc = sm_json_data.tech_json_dict[name].get('note')
    if isinstance(desc, str):
        return desc
    elif isinstance(desc, list):
        return ' '.join(desc)
    else:
        return ''


def get_tech_inputs_for_level(level, tech_list):
    if level == 'Easy':
        return ''
    level_no_space = ''.join(level.split(' '))
    rows = '\n'.join(f'''
        <div class="row">
            <div class="col-sm-3 form-check">
              <input type="checkbox" class="form-check-input" id="tech-{tech}" onchange="techChanged()" name="tech-{tech}" value="{tech}">
              <label class="form-check-label" for="{tech}"><b>{tech}</b></label>
            </div>
        </div>
        <div class="row">
            <div class="col-sm-12">
              {get_tech_description(tech)}
            </div>
        </div>
        ''' for tech in sorted(tech_list))
    return f'''
      <div class="accordion-item">
        <h2 class="accordion-header">
          <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{level_no_space}Tech">
            {level} Tech
          </button>
        </h2>
        <div id="collapse{level_no_space}Tech" class="accordion-collapse collapse">
            <div class="container mx-2 my-2">
                {rows}
            </div>
        </div>
    </div>
    '''
    # return f'''
    #   <div class="card m-2">
    #     <div class="card-header">
    #       {level} Tech
    #     </div>
    #     <div class="card-body">
    #         {rows}
    #     </div>
    #   </div>
    # '''


def get_tech_inputs():
    return '\n'.join(get_tech_inputs_for_level(level, preset['tech'])
                     for level, preset in presets)


preset_dict = {}
preset_tech_list = []
for preset_name, preset_tech in presets:
    preset_tech_list = preset_tech_list + preset_tech['tech']
    preset_dict[preset_name] = {**preset_tech, 'tech': preset_tech_list}

preset_tech_set = set(preset_tech_list)
for tech in sm_json_data.tech_name_set.difference(preset_tech_set).difference(ignored_tech_set):
    raise RuntimeError(f"Tech '{tech}' in sm-json-data but not in any preset")
for tech in preset_tech_set.difference(sm_json_data.tech_name_set):
    raise RuntimeError(f"Unrecognized tech '{tech}'")
for tech in ignored_tech_set.difference(sm_json_data.tech_name_set):
    raise RuntimeError(f"Unrecognized ignored tech '{tech}'")
for tech in preset_tech_set.intersection(ignored_tech_set):
    raise RuntimeError(f"Tech '{tech}' is ignored but marked in preset")

def item_placement_strategy_buttons():
    return '\n'.join(f'''
      <input type="radio" class="btn-check" name="itemPlacementStrategy" id="itemPlacementStrategy{name}" value="{name}" autocomplete="off" onclick="presetChanged()" {'checked' if name == 'Open' else ''}>
      <label class="btn btn-outline-primary" for="itemPlacementStrategy{name}">{name}</label>
    ''' for name in item_placement_strategies.keys())


def preset_buttons():
    return '\n'.join(f'''
      <input type="radio" class="btn-check" name="preset" id="preset{name}" autocomplete="off" onclick="presetChanged()" {'checked' if i == 0 else ''}>
      <label class="btn btn-outline-primary" for="preset{name}">{name}</label>
    ''' for i, name in enumerate(preset_dict.keys()))


def preset_change_script():
    script_list = []
    for name, preset in preset_dict.items():
        script_list.append(f'''
            if (document.getElementById("preset{name}").checked) {{
                document.getElementById("shinesparkTiles").value = {preset["shinesparkTiles"]};
                document.getElementById("resourceMultiplier").value = {preset["resourceMultiplier"]};
                document.getElementById("escapeTimerMultiplier").value = {preset["escapeTimerMultiplier"]};
                document.getElementById("saveAnimalsNo").checked = true;
                document.getElementById("saveAnimalsYes").checked = false;
                {';'.join(f'document.getElementById("tech-{tech}").checked = {"true" if tech in preset["tech"] else "false"}' for tech in preset_tech_list)}
            }}
        ''')
    return '\n'.join(script_list)


def tech_change_script():
    return '\n'.join(f'document.getElementById("preset{name}").checked = false;' for name in preset_dict.keys())


def encode_difficulty(difficulty: DifficultyConfig):
    x = 0
    x = x * 22 + (difficulty.shine_charge_tiles - 12)
    x = x * 91 + int(difficulty.resource_multiplier * 10) - 10
    for tech in sorted(sm_json_data.tech_name_set):
        x *= 2
        if tech in difficulty.tech:
            x += 1
    return x

# TODO: Use a more reasonable way of serving static content.
@app.route("/WebTitle.png")
def title_image():
    return flask.send_file("../../gfx/title/WebTitle.png", mimetype='image/png')


@app.route("/favicon.ico")
def favicon():
    return flask.send_file("../../gfx/favicon.ico", mimetype='image/png')


def change_log():
    return open('CHANGELOG.html', 'r').read()


@app.route("/")
def home():
    # TODO: Put this somewhere else instead of inline here.
    return f'''<!DOCTYPE html>
    <html lang="en-US">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Super Metroid Map Rando</title>
        <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
      </head>
      <body>
        <div class="container">
            <div class="row">
                <div class="col-9 text-center my-2">
                    <img src="/WebTitle.png" alt="Super Metroid Map Rando" style="width: 50%">
                </div>
                <div class="col-3" align=right>
                    <small>Version: {VERSION}</small>
                </div>
            </div>
            <form method="POST" enctype="multipart/form-data" action="/randomize">
                <div class="form-group row my-2">
                  <label class="col-sm-2 col-form-label" for="rom">Input ROM</label>
                  <input class="col-sm-10 form-control-file" type="file" id="rom" name="rom">
                </div>
                <div class="form-group row my-2">
                  <label class="col-sm-2 col-form-label" for="preset">Item placement</label>
                  <div class="col-sm-4 btn-group" role="group">
                    {item_placement_strategy_buttons()}
                 </div>
                </div>
                <div class="form-group row my-2 mx-2">
                    <small><strong>Open</strong>: At each step of item placement, the item will be placed at a random accessible location.
                    At the end, non-progression items such as extra E-Tanks and ammo are placed randomly across all remaining
                    locations.</small>                    
                </div>
                <div class="form-group row my-2 mx-2">
                    <small><strong>Closed</strong>: At each step of item placement, if possible the item will be placed at a random accessible location 
                    that was unlocked by the previous item. Non-progression items (except Missiles) are
                    placed in locations that become accessible as late as possible. This reduces the amount of ways to
                    progress, making it more likely that harder tech will be required.</small>                    
                </div>
                <div class="form-group row my-2">
                  <label class="col-sm-2 col-form-label" for="preset">Skill assumption</label>
                  <div class="col-sm-10 btn-group" role="group">
                    {preset_buttons()}
                 </div>
                </div>
                <div class="accordion my-2" id="accordion">
                  <div class="accordion-item">
                    <h2 class="accordion-header">
                      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDifficulty">
                        Customize skill assumptions
                      </button>
                    </h2>
                    <div id="collapseDifficulty" class="accordion-collapse collapse m-2">
                      <div class="form-group row m-2">
                        <label for="shinesparkTiles" class="col-sm-6 col-form-label">Shinespark tiles<br>
                        <small>(Smaller values assume ability to short-charge over shorter distances)</small>
                        </label>
                        <div class="col-sm-2">
                          <input type="text" class="form-control" name="shinesparkTiles" id="shinesparkTiles" value="32">
                        </div>
                      </div>
                      <div class="form-group row m-2">
                        <label for="resourceMultiplier" class="col-sm-6 col-form-label">Resource multiplier<br>
                        <small>(Leniency factor on assumed energy & ammo usage)</small>
                        </label>
                        <div class="col-sm-2">
                          <input type="text" class="form-control" name="resourceMultiplier" id="resourceMultiplier" value="3.0">
                        </div>
                      </div>
                      <div class="form-group row m-2">
                        <label for="escapeTimerMultiplier" class="col-sm-6 col-form-label">Escape timer multiplier<br>
                        <small>(Leniency factor on escape timer)</small>
                        </label>
                        <div class="col-sm-2">
                          <input type="text" class="form-control" name="escapeTimerMultiplier" id="escapeTimerMultiplier" value="3.0">
                        </div>
                      </div>
                      <div class="form-group row m-2">
                          <label class="col-sm-6 col-form-label" for="preset">Save the animals<br>
                          <small>(Take into account extra time needed in the escape)</small></label>
                          <div class="col-sm-4 btn-group my-3" role="group">
                                <input type="radio" class="btn-check" name="saveAnimals" id="saveAnimalsNo" value="No" checked=true>
                                <label class="btn btn-outline-primary" for="saveAnimalsNo">No</label>
                                <input type="radio" class="btn-check" name="saveAnimals" id="saveAnimalsYes" value="Yes">
                                <label class="btn btn-outline-primary" for="saveAnimalsYes">Yes</label>
                          </div>
                      </div>
                      <div class="form-group row my-2">
                        <div class="col-sm-12">
                          <div class="accordion my-2" id="accordion">
                            {get_tech_inputs()}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="form-group row my-2">
                  <label for="randomSeed" class="col-sm-2 col-form-label">Random seed</label>
                  <div class="col-sm-3">
                    <input type="text" class="form-control" name="randomSeed" id="randomSeed" value="">
                  </div>
                </div>
                <div class="form-group row my-2">
                    <div class="col-sm-12">
                        <input type="submit" class="btn btn-primary" value="Generate ROM" />
                    </div>
                </div>
            </form>
            <small><strong>Note</strong>: ROM may take a while to generate. For fastest results, click "Generate ROM" only once and wait patiently. 
            If it times out, try again with a different random seed. This is still in an alpha stage of development, 
            so bugs are expected. If you encounter a problem, 
            feedback is welcome on <a href="https://github.com/blkerby/MapRandomizer/issues">GitHub issues</a>. 
            Also feel free to stop by the <a href="https://discord.gg/Gc99YV2ZcB">Discord</a>: let us know if you
            find a cool seed (or a broken seed), if you have questions or ideas for future development, or if you're 
            streaming the game!</small>
            <div class="row my-2">
                <div class="col-sm-12">
                    <div class="card">
                        <div class="card-header">Things to know</div>
                        <div class="card-body">
                            <ul>
                            <li>Certain items do not spawn until the planet is 
                            awakened, by exiting Pit Room (old Mother Brain
                            room) with Morph and Missiles collected.<a href="#footnote-pit-room"><sup>1</sup></a>
                            <li>Certain items do not spawn until Phantoon has been
                            defeated.
                            <li>Phantoon's Room is always two rooms away from the Wrecked Ship Map Room, and both are 
                                in the same area as the Wrecked Ship Save Room.
                            <li>Items are always marked by dots on the map. Map stations, refills, 
                            and major bosses (G4 and Mother Brain) are marked by special tiles.
                            <li>Missile Refill stations refill all ammo types: Missiles, Supers, and Power Bombs.
                            <li>Gravity and Varia behave like Progressive Suits in other randomizers,
                            each giving 50% reduction in enemy damage (stacking to a combined 75%).
                            <li>Mother Brain has been changed to take double damage from Supers.
                            <li>The current tile can be marked un-explored on the map  
                            by pressing Angle Up and Item Cancel simultaneously. To be effective, these inputs must be
                            held while exiting the tile, since otherwise the game will immediately re-explore the tile.
                            <li>Saving at a different save station from the last save will advance to the next slot 
                            before saving, so you can return to an earlier save in case you get stuck.
                            <li>Samus collects & equips all items (excluding beams, ammo, and tanks) when acquiring Hyper Beam.
                            </ul>
                            <p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row my-2">
                <div class="col-sm-12">
                    <div class="card">
                        <div class="card-header">Known issues</div>
                        <div class="card-body">
                            <ul>
                            <li>Even if the tech is not selected, wall jumps and crouch-jump/down-grabs may be required in some places.
                            <li>On Closed settings the game tends to be very stingy with giving extra ammo/tanks (other than Missiles).
                            <li>Some sound effects are glitched (due to changing the music).
                            <li>Some map tiles associated with elevators do not appear correctly.
                            <li>Door transitions generally have some minor graphical glitches.
                            <li>The end credits are vanilla.
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row my-2">
                <div class="col-sm-12">
                    <div class="card">
                        <div class="card-header">Change log</div>
                        <div class="card-body">
                            {change_log()}
                        </div>
                    </div>
                </div>
            </div>
            <div class="row my-2">
                <div class="col-sm-12">
                    <a id="footnote-pit-room">
                    <sup>1</sup> Technically the planet is awakened by opening any gray door locked by an enemy kill 
                    count (which does <i>not</i> include gray doors locked by boss kills).
                    In the vanilla game, such gray doors spawn in the Pit Room when entering with Morph and Missiles but
                    are also in other rooms (e.g., Spore Spawn Kihunters, Brinstar Pre-Map Room). In 
                    Map Rando, at least for now, all gray doors are removed except in the Pit Room, making the Pit Room
                    the only place where the planet can be awakened. Note that Pit Room also has an item that only
                    spawns when entering with Morph and Missiles (regardless of whether the planet is awake).
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>        
        <script>
        document.getElementById("randomSeed").value = Math.floor(Math.random() * 0x7fffffff);        
        function presetChanged() {{
            {preset_change_script()}
        }}
        function techChanged() {{
            {tech_change_script()}
        }}
        </script>
      </body>
    </html>
    '''


@app.route("/randomize", methods=['POST'])
def randomize():
    uploaded_rom_file = flask.request.files['rom']
    input_buf = uploaded_rom_file.stream.read(10_000_000)

    if len(input_buf) < 3145728 or len(input_buf) > 8388608:
        return flask.Response("Invalid input ROM", status=400)

    try:
        item_placement_strategy = item_placement_strategies[flask.request.form.get('itemPlacementStrategy')]
    except:
        return flask.Response("Invalid itemPlacementStrategy: '{}'".format(flask.request.form.get('itemPlacementStrategy')))

    try:
        shine_charge_tiles = int(flask.request.form.get('shinesparkTiles'))
        assert shine_charge_tiles >= 12 and shine_charge_tiles <= 33
    except:
        return flask.Response("Invalid shinesparkTiles", status=400)

    try:
        resource_multiplier = float(flask.request.form.get('resourceMultiplier'))
        resource_multiplier = round(resource_multiplier * 10) / 10
        assert 1.0 <= resource_multiplier <= 10.0
    except:
        return flask.Response("Invalid resourceMultiplier", status=400)

    try:
        escape_timer_multiplier = float(flask.request.form.get('escapeTimerMultiplier'))
        assert 0.0 < escape_timer_multiplier <= 10.0
    except:
        return flask.Response("Invalid escapeTimerMultiplier", status=400)

    try:
        save_animals = flask.request.form.get('saveAnimals') == 'Yes'
    except:
        return flask.Response("Invalid escapeTimerMultiplier", status=400)

    try:
        random_seed = int(flask.request.form.get('randomSeed'))
    except:
        return flask.Response("Invalid randomSeed", status=400)

    tech = set(tech for tech in sm_json_data.tech_name_set if flask.request.form.get('tech-' + tech) != None)
    difficulty = DifficultyConfig(tech=tech, shine_charge_tiles=shine_charge_tiles,
                                  resource_multiplier=resource_multiplier,
                                  escape_time_multiplier=escape_timer_multiplier,
                                  save_animals=save_animals)
    output_file_prefix = f'smmr-v{VERSION}-{random_seed}-{encode_difficulty(difficulty)}'
    logging.info(f"Starting {output_file_prefix}: random_seed={random_seed}, item_placement_strategy={item_placement_strategy}, difficulty={difficulty}, ROM='{uploaded_rom_file.filename}' (hash={hash(input_buf)})")
    max_map_attempts = 500
    max_item_attempts = 1
    np.random.seed(random_seed % (2 ** 32))
    random.seed(random_seed)

    for _ in range(max_map_attempts):
        map_i = int(np.random.randint(0, len(file_list)))
        map_filename = file_list[map_i]
        map_file = '{}/{}'.format(map_dir, map_filename)
        map = json.load(open(map_file, 'r'))
        logging.info("{}".format(map_file))

        # Switch around single-tile rooms to balance the distribution of utility rooms (maps, saves, refills)
        map = balance_utilities(map)
        if map is None:
            continue

        randomizer = Randomizer(map, sm_json_data, difficulty)
        for i in range(max_item_attempts):
            success = randomizer.randomize(item_placement_strategy)
            if success:
                break
        else:
            continue
        break
    else:
        return flask.Response("Too many failed item randomization attempts", status=500)

    logging.info("Done with item randomization")
    spoiler_items = []
    for i in range(len(randomizer.item_placement_list)):
        spoiler_items.append({
            'nodeAddress': '{:X}'.format(randomizer.item_placement_list[i]),
            'item': randomizer.item_sequence[i],
        })

    config = {
        'version': VERSION,
        'seed': random_seed,
        'item_placement_strategy': item_placement_strategy.value,
        'shine_charge_tiles': difficulty.shine_charge_tiles,
        'resource_multiplier': difficulty.resource_multiplier,
        'escape_time_multiplier': difficulty.escape_time_multiplier,
        'save_animals': difficulty.save_animals,
        'tech': list(sorted(difficulty.tech)),
    }

    # Rerank the areas to assign the less nice music to smaller areas:
    map = rerank_areas(map)

    xs_min = np.array([p[0] for p in map['rooms']])
    ys_min = np.array([p[1] for p in map['rooms']])
    xs_max = np.array([p[0] + rooms[i].width for i, p in enumerate(map['rooms'])])
    ys_max = np.array([p[1] + rooms[i].height for i, p in enumerate(map['rooms'])])

    door_room_dict = {}
    for i, room in enumerate(rooms):
        for door in room.door_ids:
            door_pair = (door.exit_ptr, door.entrance_ptr)
            door_room_dict[door_pair] = i
    edges_list = []
    for conn in map['doors']:
        src_room_id = door_room_dict[tuple(conn[0])]
        dst_room_id = door_room_dict[tuple(conn[1])]
        edges_list.append((src_room_id, dst_room_id))

    room_graph = graph_tool.Graph(directed=True)
    for (src, dst) in edges_list:
        room_graph.add_edge(src, dst)
        room_graph.add_edge(dst, src)

    num_areas = 6
    area_arr = np.array(map['area'])

    # Ensure that Landing Site is in Crateria:
    area_arr = (area_arr - area_arr[1] + num_areas) % num_areas

    display = MapDisplay(72, 72, 20)
    display.display_assigned_areas(map)
    # display.display_assigned_areas_with_maps(map)
    # display.display_assigned_areas_with_saves(map)
    # display.display_assigned_areas_with_ws(map)
    map_png_file = io.BytesIO()
    display.image.save(map_png_file, "png")
    map_png_bytes = map_png_file.getvalue()

    display = MapDisplay(72, 72, 20)
    display.display_vanilla_areas(map)
    map_orig_png_file = io.BytesIO()
    display.image.save(map_orig_png_file, "png")
    map_orig_png_bytes = map_orig_png_file.getvalue()


    orig_rom = Rom(io.BytesIO(input_buf))
    rom = Rom(io.BytesIO(input_buf))

    # Patches to be applied at beginning (before reconnecting doors, etc.)
    orig_patches = [
        'mb_barrier2',
        'mb_barrier_clear',
        'hud_expansion_opaque',
        'gray_doors',
    ]
    for patch_name in orig_patches:
        patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
        orig_rom.bytes_io = BytesIO(patch.apply(orig_rom.bytes_io.getvalue()))
        rom.bytes_io = BytesIO(patch.apply(rom.bytes_io.getvalue()))

    # Change Aqueduct map y position, to include the toilet (for the purposes of the map)
    old_y = orig_rom.read_u8(0x7D5A7 + 3)
    orig_rom.write_u8(0x7D5A7 + 3, old_y - 4)

    # Change door asm for entering mother brain room from right
    # orig_rom.write_u16(0x1AAC8 + 10, 0xEB00)
    # # rom.write_u16(0x1956A + 10, 0xEB00)

    rom.write_u8(snes2pc(0x83AA8F), 0x04)  # Stop wall from spawning in Tourian Escape Room 1: door direction = 4 (right)

    # Area data: --------------------------------
    area_index_dict = defaultdict(lambda: {})
    for i, room in enumerate(rooms):
        orig_room_area = orig_rom.read_u8(room.rom_address + 1)
        room_index = orig_rom.read_u8(room.rom_address)
        assert room_index not in area_index_dict[orig_room_area]
        area_index_dict[orig_room_area][room_index] = area_arr[i]
    # Handle twin rooms
    aqueduct_room_i = [i for i, room in enumerate(rooms) if room.name == 'Aqueduct'][0]
    area_index_dict[4][0x18] = area_arr[aqueduct_room_i]  # Set Toilet to same area as Aqueduct
    pants_room_i = [i for i, room in enumerate(rooms) if room.name == 'Pants Room'][0]
    area_index_dict[4][0x25] = area_arr[pants_room_i]  # Set East Pants Room to same area as Pants Room
    west_ocean_room_i = [i for i, room in enumerate(rooms) if room.name == 'West Ocean'][0]
    area_index_dict[0][0x11] = area_arr[west_ocean_room_i]  # Set Homing Geemer Room to same area as West Ocean

    # Write room map area data: we use free space in the ROM to store the map area for each room. We leave the area in
    # the room header alone because changing that would break a bunch of things.
    area_sizes = [max(area_index_dict[i].keys()) + 1 for i in range(num_areas)]
    cumul_area_sizes = [0] + list(np.cumsum(area_sizes))
    area_data_base_ptr = 0x7E99B  # LoRom $8F:E99B
    area_data_ptrs = [area_data_base_ptr + num_areas * 2 + cumul_area_sizes[i] for i in range(num_areas)]
    assert area_data_ptrs[-1] <= 0x7EB00
    for i in range(num_areas):
        rom.write_u16(area_data_base_ptr + 2 * i, (area_data_ptrs[i] & 0x7FFF) + 0x8000)
        for room_index, new_area in area_index_dict[i].items():
            rom.write_u8(area_data_ptrs[i] + room_index, new_area)

    # print("{:x}".format(area_data_ptrs[-1] + area_sizes[-1]))

    # Write map data:
    # first clear existing maps
    for area_id, area_ptr in area_map_ptrs.items():
        for i in range(64 * 32):
            # if area_id == 0:
            #     rom.write_u16(area_ptr + i * 2, 0x0C1F)
            # else:
            rom.write_u16(area_ptr + i * 2, 0x001F)

    area_start_x = []
    area_start_y = []
    for i in range(num_areas):
        ind = np.where(area_arr == i)
        area_start_x.append(np.min(xs_min[ind]) - 2)
        area_start_y.append(np.min(ys_min[ind]) - 1)
    for i, room in enumerate(rooms):
        rom_room = RomRoom(orig_rom, room)
        area = area_arr[i]
        rom_room.area = area
        rom_room.x = xs_min[i] - area_start_x[area]
        rom_room.y = ys_min[i] - area_start_y[area]
        rom_room.write_map_data(rom)
        if room.name == 'Aqueduct':
            # Patch map tile in Aqueduct to replace Botwoon Hallway with tube/elevator tile
            # TODO: move this in with the other map patches.
            cell = rom.read_u16(rom_room.xy_to_map_ptr(rom_room.x + 2, rom_room.y + 2))
            rom.write_u16(rom_room.xy_to_map_ptr(rom_room.x + 2, rom_room.y + 3), cell)

    door_room_dict = {}
    for i, room in enumerate(rooms):
        for door_id in room.door_ids:
            door_room_dict[(door_id.exit_ptr, door_id.entrance_ptr)] = i

    # Find the rooms connected to Kraid and Crocomire and set them to reload CRE (to prevent graphical glitches).
    # Not sure if this is necessary for Crocomire, but the vanilla game does this so we do it to be safe.
    reload_cre_door_pairs = [
        (0x191DA, 0x19252),  # Kraid right door
        (0x191CE, 0x191B6),  # Kraid left door
        (0x193DE, 0x19432),  # Crocomire left door
        (0x193EA, 0x193D2),  # Crocomire top door
    ]
    for src, dst, _ in map['doors']:
        if tuple(src) in reload_cre_door_pairs:
            dst_room_i = door_room_dict[tuple(dst)]
            rom.write_u8(rooms[dst_room_i].rom_address + 8, 2)  # Special GFX flag = Reload CRE
        if tuple(dst) in reload_cre_door_pairs:
            src_room_i = door_room_dict[tuple(src)]
            rom.write_u8(rooms[src_room_i].rom_address + 8, 2)  # Special GFX flag = Reload CRE

    boss_exit_asm = 0xF7F0
    toilet_exit_asm = 0xE301  # Return control of Samus after exiting the toilet
    JSR = lambda x: bytes([0x20, x & 0xFF, x >> 8])

    def explore_map_tile_asm(area, x, y):
        byte_addr, bitmask = get_area_explored_bit_ptr(area, x, y)
        out = bytearray()
        out.extend([0xBF, byte_addr & 0xFF, byte_addr >> 8, 0x7E])  # LDA $7E:{byte_addr}
        # out.extend([0x09, bitmask, 0x00])  # ORA #{bitmask}
        out.extend([0x09, 0xFF, 0xff])  # ORA #{bitmask}
        out.extend([0x9F, byte_addr & 0xFF, byte_addr >> 8, 0x7E])  # STA $7E:{byte_addr}
        return bytes(out)

    extra_door_asm_dict = {
        0x1A600: JSR(toilet_exit_asm),  # Aqueduct door down
        0x1A60C: JSR(toilet_exit_asm),  # Aqueduct door up
        0x191CE: JSR(boss_exit_asm),  # Kraid left
        0x191DA: JSR(boss_exit_asm),  # Kraid right
        0x1A96C: JSR(boss_exit_asm),  # Draygon right
        0x1A978: JSR(boss_exit_asm),  # Draygon left
        0x193DE: JSR(boss_exit_asm),  # Crocomire left door
        0x193EA: JSR(boss_exit_asm),  # Crocomire top door
        0x1A2C4: JSR(boss_exit_asm),  # Phantoon door
        0x18916: bytes([0xCE, 0xC6, 0x09]) + b''.join(explore_map_tile_asm(0, i, i) for i in range(30)),  # Landing site door (testing)
    }
    extra_door_asm_location = {}
    door_asm_free_space = 0xED00  # in bank $8F
    for door_addr, extra_asm_bytes in extra_door_asm_dict.items():
        extra_door_asm_location[door_addr] = door_asm_free_space
        door_asm_free_space += len(extra_asm_bytes) + 3  # Reserve 3 bytes for the JMP instruction to the original ASM

    def write_door_data(ptr, data):
        # print("door: {:x}: {}".format(ptr, data.tobytes()))
        rom.write_n(ptr, 12, data)
        bitflag = data[2] | 0x40
        rom.write_u8(ptr + 2, bitflag)

        # When leaving the aqueduct and entering a door that normally has a door ASM into it, we use a patched door
        # ASM to run both the original door ASM (for leaving the toilet) and the door ASM for the destination room.
        # Both are essential: if we don't run the original toilet door ASM, then Samus will be left frozen, unable to
        # be controlled; if we don't run the door ASM for the next room, we can end up with nasty glitches, e.g. due to
        # camera scrolls not being set. Note, for the toilet door ASM, we use the northbound version for both doors
        # since the southbound one sets camera scrolls (for Oasis) that generally won't be applicable in the next
        # room. Using a similar technique we also run extra ASM for exiting bosses to prevent graphical glitches.
        if ptr in extra_door_asm_dict:
            extra_asm = extra_door_asm_dict[ptr]
            free_space = extra_door_asm_location[ptr]
            # Create a new ASM in free space to run both the extra door ASM and destination door ASM (if applicable).
            rom.write_u16(ptr + 10, free_space)
            rom.write_n(0x70000 + free_space, len(extra_asm), extra_asm)
            if data[10] != 0 or data[11] != 0:
                rom.write_u8(0x70000 + free_space + len(extra_asm), 0x4C)  # JMP opcode (Jump)
                rom.write_n(0x70000 + free_space + len(extra_asm) + 1, 2, data[10:12])  # Run the door ASM for next room
            else:
                # Return, because there is no original destination door ASM to run.
                rom.write_u8(0x70000 + free_space + len(extra_asm), 0x60)  # RTS opcode (return from subroutine)
        elif ptr == 0x1A798:  # Pants Room right door
            rom.write_n(0x1A7BC, 12, data)  # Also write the same data to the East Pants Room right door
            rom.write_u8(0x1A7BC + 2, bitflag)

    def write_door_connection(a, b):
        a_exit_ptr, a_entrance_ptr = a
        b_exit_ptr, b_entrance_ptr = b
        if a_entrance_ptr is not None and b_exit_ptr is not None:
            # print('{:x},{:x}'.format(a_entrance_ptr, b_exit_ptr))
            a_entrance_data = orig_rom.read_n(a_entrance_ptr, 12)
            write_door_data(b_exit_ptr, a_entrance_data)
            # rom.write_n(b_exit_ptr, 12, a_entrance_data)
        if b_entrance_ptr is not None and a_exit_ptr is not None:
            b_entrance_data = orig_rom.read_n(b_entrance_ptr, 12)
            write_door_data(a_exit_ptr, b_entrance_data)
            # rom.write_n(a_exit_ptr, 12, b_entrance_data)
            # print('{:x} {:x}'.format(b_entrance_ptr, a_exit_ptr))

    for (a, b, _) in list(map['doors']):
        write_door_connection(a, b)

    save_station_ptrs = [
        0x44C5,
        0x44D3,
        0x45CF,
        0x45DD,
        0x45EB,
        0x45F9,
        0x4607,
        0x46D9,
        0x46E7,
        0x46F5,
        0x4703,
        0x4711,
        0x471F,
        0x481B,
        0x4917,
        0x4925,
        0x4933,
        0x4941,
        0x4A2F,
        0x4A3D,
    ]

    area_save_ptrs = [0x44C5, 0x45CF, 0x46D9, 0x481B, 0x4917, 0x4A2F]

    orig_door_dict = {}
    for room in rooms:
        for door in room.door_ids:
            orig_door_dict[door.exit_ptr] = door.entrance_ptr
            # if door.exit_ptr is not None:
            #     door_asm = orig_rom.read_u16(door.exit_ptr + 10)
            #     if door_asm != 0:
            #         print("{:x}".format(door_asm))

    door_dict = {}
    for (a, b, _) in map['doors']:
        a_exit_ptr, a_entrance_ptr = a
        b_exit_ptr, b_entrance_ptr = b
        if a_exit_ptr is not None and b_exit_ptr is not None:
            door_dict[a_exit_ptr] = b_exit_ptr
            door_dict[b_exit_ptr] = a_exit_ptr

    # Fix save stations
    for ptr in save_station_ptrs:
        orig_entrance_door_ptr = orig_rom.read_u16(ptr + 2) + 0x10000
        exit_door_ptr = orig_door_dict[orig_entrance_door_ptr]
        entrance_door_ptr = door_dict[exit_door_ptr]
        rom.write_u16(ptr + 2, entrance_door_ptr & 0xffff)
    #
    # # Fix save stations
    # room_ptr_to_idx = {room.rom_address: i for i, room in enumerate(rooms)}
    # area_save_idx = {x: 0 for x in range(6)}
    # area_save_idx[0] = 1  # Start Crateria index at 1 since we keep ship save station as is.
    # for ptr in save_station_ptrs:
    #     room_ptr = orig_rom.read_u16(ptr) + 0x70000
    #     if room_ptr != 0x791F8:  # The ship has no Save Station PLM for us to update (and we don't need to since we keep the ship in Crateria)
    #         room_obj = Room(orig_rom, room_ptr)
    #         states = room_obj.load_states(orig_rom)
    #         plm_ptr = states[0].plm_set_ptr + 0x70000
    #         plm_type = orig_rom.read_u16(plm_ptr)
    #         assert plm_type == 0xB76F  # Check that the first PLM is a save station
    #
    #         area = cs[room_ptr_to_idx[room_ptr]]
    #         idx = area_save_idx[area]
    #         rom.write_u16(plm_ptr + 4, area_save_idx[area])
    #         area_save_idx[area] += 1
    #
    #         orig_save_station_bytes = orig_rom.read_n(ptr, 14)
    #         dst_ptr = area_save_ptrs[area] + 14 * idx
    #         rom.write_n(dst_ptr, 14, orig_save_station_bytes)
    #     else:
    #         area = 0
    #         dst_ptr = ptr
    #
    #     orig_entrance_door_ptr = rom.read_u16(dst_ptr + 2) + 0x10000
    #     exit_door_ptr = orig_door_dict[orig_entrance_door_ptr]
    #     entrance_door_ptr = door_dict[exit_door_ptr] & 0xffff
    #     rom.write_u16(dst_ptr + 2, entrance_door_ptr & 0xffff)

    # item_dict = {}
    plm_types_to_remove = [
        0xC88A, 0xC85A, 0xC872,  # right pink/yellow/green door
        0xC890, 0xC860, 0xC878,  # left pink/yellow/green door
        0xC896, 0xC866, 0xC87E,  # down pink/yellow/green door
        0xC89C, 0xC86C, 0xC884,  # up pink/yellow/green door
        0xDB48, 0xDB4C, 0xDB52, 0xDB56, 0xDB5A, 0xDB60,  # eye doors
        0xC8CA,  # wall in Escape Room 1
    ]
    gray_door_plm_types = [
        0xC848,  # left gray door
        0xC842,  # right gray door
        0xC854,  # up gray door
        0xC84E,  # down gray door
    ]
    keep_gray_door_room_names = [
        "Pit Room",
        "Kraid Room",
        "Draygon's Room",
        "Ridley's Room",
        "Golden Torizo's Room",
    ]
    for room_obj in rooms:
        room = RomRoom(orig_rom, room_obj)
        states = room.load_states(rom)
        for state in states:
            ptr = state.plm_set_ptr + 0x70000
            while True:
                plm_type = orig_rom.read_u16(ptr)
                if plm_type == 0:
                    break
                # Remove PLMs for doors that we don't want: pink, green, yellow, Eye doors, spawning wall in escape
                # main_var_high = orig_rom.read_u8(ptr + 5)
                # is_removable_gray_door = plm_type in gray_door_plm_types and main_var_high != 0x0C and room_obj.name not in keep_gray_door_room_names
                is_removable_gray_door = plm_type in gray_door_plm_types and room_obj.name not in keep_gray_door_room_names
                if plm_type in plm_types_to_remove or is_removable_gray_door:
                    rom.write_u16(ptr, 0xB63B)  # right continuation arrow (should have no effect, giving a blue door)
                    rom.write_u16(ptr + 2, 0)  # position = (0, 0)
                ptr += 6

    def item_to_plm_type(item_name, orig_plm_type):
        item_list = [
            "ETank",
            "Missile",
            "Super",
            "PowerBomb",
            "Bombs",
            "Charge",
            "Ice",
            "HiJump",
            "SpeedBooster",
            "Wave",
            "Spazer",
            "SpringBall",
            "Varia",
            "Gravity",
            "XRayScope",
            "Plasma",
            "Grapple",
            "SpaceJump",
            "ScrewAttack",
            "Morph",
            "ReserveTank",
        ]
        i = item_list.index(item_name)
        old_i = ((orig_plm_type - 0xEED7) // 4) % 21
        return orig_plm_type + (i - old_i) * 4

    # Place items
    for i in range(len(randomizer.item_placement_list)):
        ptr = randomizer.item_placement_list[i]
        item_name = randomizer.item_sequence[i]
        orig_plm_type = orig_rom.read_u16(ptr)
        plm_type = item_to_plm_type(item_name, orig_plm_type)
        rom.write_u16(ptr, plm_type)

    # Copy the item at Morph Ball to the Zebes-awake state (so it doesn't become unobtainable after Zebes is awake).
    # For this we overwrite the PLM slot for the gray door at the left of the room (which we would get rid of anyway).
    rom.write_n(0x78746, 6, rom.read_n(0x786DE, 6))

    map_patcher = MapPatcher(rom, area_arr)
    map_patcher.apply_map_patches()
    map_patcher.add_cross_area_arrows(map)
    map_patcher.set_map_stations_explored(map)

    # print(randomizer.item_sequence[:5])
    # print(randomizer.item_placement_list[:5])
    # sm_json_data.node_list[641]

    # # Randomize items
    # item_list = list(item_dict.values())
    # item_perm = np.random.permutation(len(item_dict.values()))
    # for i, ptr in enumerate(item_dict.keys()):
    #     item = item_list[item_perm[i]]
    #     rom.write_u16(ptr, item)

    # rom.write_u16(0x78000, 0xC82A)
    # rom.write_u8(0x78002, 40)
    # rom.write_u8(0x78003, 68)
    # rom.write_u16(0x78004, 0x8000)

    # ---- Fix twin room map x & y:
    # Aqueduct:
    old_aqueduct_x = rom.read_u8(0x7D5A7 + 2)
    old_aqueduct_y = rom.read_u8(0x7D5A7 + 3)
    rom.write_u8(0x7D5A7 + 3, old_aqueduct_y + 4)
    # Toilet:
    rom.write_u8(0x7D408 + 2, old_aqueduct_x + 2)
    rom.write_u8(0x7D408 + 3, old_aqueduct_y)
    # East Pants Room:
    pants_room_x = rom.read_u8(0x7D646 + 2)
    pants_room_y = rom.read_u8(0x7D646 + 3)
    rom.write_u8(0x7D69A + 2, pants_room_x + 1)
    rom.write_u8(0x7D69A + 3, pants_room_y + 1)
    # Homing Geemer Room:
    west_ocean_x = rom.read_u8(0x793FE + 2)
    west_ocean_y = rom.read_u8(0x793FE + 3)
    rom.write_u8(0x7968F + 2, west_ocean_x + 5)
    rom.write_u8(0x7968F + 3, west_ocean_y + 2)

    # Write palette and tilemap for title background:
    import PIL
    import PIL.Image
    title_bg_png = PIL.Image.open('gfx/title/Title3.png')
    # title_bg_png = PIL.Image.open('gfx/title/titlesimplified2.png')
    title_bg = np.array(title_bg_png)[:, :, :3]
    pal, gfx, tilemap = encode_graphics(title_bg)
    gfx = gfx + 1  # Avoid touching the background color (0)
    compressed_gfx = compress(gfx.tobytes())
    compressed_tilemap = compress(tilemap.tobytes())
    # print("Compressed GFX size:", len(compressed_gfx))
    # print("Compressed tilemap size:", len(compressed_tilemap))
    rom.write_n(0x661E9 + 2, len(pal.tobytes()), pal.tobytes())
    # Use white color for Nintendo copyright text (otherwise it would stay black since we skip the palette FX handler)
    rom.write_u16(0x661E9 + 0xC9 * 2, 0x7FFF)
    gfx_free_space_pc = 0x1C0000
    gfx_free_space_snes = pc2snes(gfx_free_space_pc)
    # rom.write_n(0xA6000, len(compressed_gfx), compressed_gfx)
    rom.write_n(gfx_free_space_pc, len(compressed_gfx), compressed_gfx)
    rom.write_u8(snes2pc(0x8B9BA8), gfx_free_space_snes >> 16)
    rom.write_u16(snes2pc(0x8B9BAC), gfx_free_space_snes & 0xFFFF)

    gfx_free_space_pc += len(compressed_gfx)
    gfx_free_space_snes = pc2snes(gfx_free_space_pc)
    rom.write_n(gfx_free_space_pc, len(compressed_tilemap), compressed_tilemap)
    rom.write_u8(snes2pc(0x8B9BB9), gfx_free_space_snes >> 16)
    rom.write_u16(snes2pc(0x8B9BBD), gfx_free_space_snes & 0xFFFF)
    # rom.write_n(snes2pc(0x8B9CB6), 3, bytes([0xEA, 0xEA, 0xEA]))  # Skip spawning baby metroid (NOP:NOP:NOP)
    # rom.write_u8(snes2pc(0x8B97F7), 0x60)  # Skip spawn text glow
    rom.write_n(snes2pc(0x8B9A34), 4, bytes([0xEA, 0xEA, 0xEA, 0xEA]))  # Skip pallete FX handler
    # rom.write_n(0xB7C04, len(compressed_tilemap), compressed_tilemap)

    gfx_free_space_pc += len(compressed_tilemap)
    gfx_free_space_snes = pc2snes(gfx_free_space_pc)
    add_title(rom, gfx_free_space_snes)

    # Apply patches
    patches = [
        'vanilla_bugfixes',
        'new_game_extra' if args.debug else 'new_game',
        'music',
        'crateria_sky_fixed',
        'everest_tube',
        'sandfalls',
        'saveload',
        'map_area',
        'elevators_speed',
        'boss_exit',
        'itemsounds',
        'progressive_suits',
        'disable_map_icons',
        'escape',
        'mother_brain_no_drain',
        'tourian_map',
        'tourian_eye_door',
        'no_explosions_before_escape',
        'escape_room_1',
        'unexplore',
        'max_ammo_display',
        'missile_refill_all',
        'sound_effect_disables',
        'title_map_animation',
    ]
    for patch_name in patches:
        patch = ips_util.Patch.load('patches/ips/{}.ips'.format(patch_name))
        rom.bytes_io = BytesIO(patch.apply(rom.bytes_io.getvalue()))

    patch_music(rom, map)

    # rom.write_u16(0x79213 + 24, 0xEB00)
    # rom.write_u16(0x7922D + 24, 0xEB00)
    # rom.write_u16(0x79247 + 24, 0xEB00)
    # rom.write_u16(0x79247 + 24, 0xEB00)
    # rom.write_u16(0x79261 + 24, 0xEB00)

    # # Connect bottom left landing site door to mother brain room, for testing
    # if args.debug:
    #     mb_door_bytes = orig_rom.read_n(0X1AAC8, 12)
    #     rom.write_n(0x18916, 12, mb_door_bytes)

    # Restore acid in Tourian Escape Room 4:
    rom.write_u16(snes2pc(0x8FDF03), 0xC953)  # Vanilla setup ASM pointer
    rom.write_u8(snes2pc(0x8FC95B), 0x60)  # RTS (return early from setup ASM to skip setting up shaking)

    # # Change setup asm for Mother Brain room
    # rom.write_u16(0x7DD6E + 24, 0xEB00)


    # title_bg_pal = open('gfx/title/title_bg.pal', 'rb').read()
    # rom.write_n(0x661E9, 512, title_bg_pal)
    # rom.write_n(0x663E9, 512, title_bg_pal)
    # rom.write_n(0x665E9, 512, title_bg_pal)
    # # title_bg_map = open('gfx/title/title_bg.m7', 'rb').read()
    # title_bg_map = open('gfx/title/title_bg.map', 'rb').read()
    # # title_bg_gfx_addr = 0xA6000
    # title_bg_map_addr = 0xB7C04
    #
    # # rom.write_u8(title_bg_map_addr, 0xFF)
    # # addr = title_bg_map_addr
    # for i in range(15):
    #     rom.write_u8(title_bg_map_addr + i * 33, 0x1F)
    #     rom.write_n(title_bg_map_addr + i * 33 + 1, 32, title_bg_map[(i * 32):((i + 1) * 32)])
    #     rom.write_u8(title_bg_map_addr + i * 33 + 33, 0xFF)

    # Set up door-specific FX:
    door_fx = {
        (0x19732, 0x1929A): 0x8386D0,  # Rising Tide left door: lava rising
        (0x1965A, 0x19672): 0x838650,  # Volcano Room left door: lava rising
        # (0x18B6E, 0x1AB34): 0x838060,  # Climb bottom-left door: lava rising
        (0x195B2, 0x195BE): 0x8385E0,  # Speed Booster Hall right door: lava rising when Speed Booster collected
        (0x1983A, 0x19876): 0x83876A,  # Acid Statue Room bottom-right door: acid lowered
        (0x199A2, 0x199F6): 0x83883C,  # Amphitheatre right door: acid raised
    }

    # In vanilla game, lava will rise in Climb if entered through Tourian Escape Room 4 (even if Zebes not ablaze).
    # Prevent this by replacing the Tourian Escape Room 4 door with the value 0xFFFF which does not match any door:
    rom.write_u16(snes2pc(0x838060), 0xffff)

    for door1, door2, _ in map['doors']:
        door1_t = tuple(door1)
        door2_t = tuple(door2)
        if door1_t in door_fx:
            # print("door1: {:x} {:x}".format(door1[0], door1[1]))
            rom.write_u16(snes2pc(door_fx[door1_t]), door2[0] & 0xffff)
        elif door2_t in door_fx:
            # print("door2: {:x} {:x}".format(door2[0], door2[1]))
            rom.write_u16(snes2pc(door_fx[door2_t]), door1[0] & 0xffff)

    # In Crocomire's initialization, skip setting the leftmost screens to red scroll. Even in the vanilla game there
    # is no purpose to this, as they are already red. But it important to skip here in the rando, because when entering
    # from the left door with Crocomire still alive, these scrolls are set to blue by the door ASM, and if they
    # were overridden with red it would break the game.
    rom.write_n(snes2pc(0xA48A92), 4, bytes([0xEA, 0xEA, 0xEA, 0xEA]))  # NOP:NOP:NOP:NOP

    # Disable demo (by overwriting the branch on the timer reaching zero):
    rom.write_n(snes2pc(0x8B9F2C), 2, bytes([0x80, 0x0A]))  # BRA $0A

    # Release Spore Spawn camera so it won't be glitched when entering from the right.
    rom.write_n(snes2pc(0xA5EADA), 3, bytes([0xEA, 0xEA, 0xEA]))  # NOP:NOP:NOP

    # TODO: Likewise release Kraid camera so it won't be as glitched when entering from the right.
    # rom.write_u16(snes2pc(0xA7A9E5), 0x0000)
    # rom.write_n(snes2pc(0xA7A9E7), 3, bytes([0xEA, 0xEA, 0xEA]))  # NOP:NOP:NOP
    # rom.write_n(snes2pc(0xA7A9ED), 4, bytes([0xEA, 0xEA, 0xEA, 0xEA]))  # NOP:NOP:NOP:NOP
    rom.write_n(snes2pc(0xA7A9F4), 4, bytes([0xEA, 0xEA, 0xEA, 0xEA]))  # NOP:NOP:NOP:NOP
    rom.write_u8(snes2pc(0xA7C9EE), 0x60)  # RTS. No longer restrict Samus X position to left screen

    # In Shaktool room, skip setting screens to red scroll (so that it won't glitch out when entering from the right):
    rom.write_u8(snes2pc(0x84B8DC), 0x60)  # RTS

    # Remove the wall that appears on the right side of Tourian Escape Room 1. This is probably redundant with the
    # door data change above. (TODO: Verify and remove this.)
    rom.write_u16(snes2pc(0x84BB34), 0x86BC)
    rom.write_u16(snes2pc(0x84BB44), 0x86BC)

    # Make Supers do double damage to Mother Brain.
    rom.write_u8(snes2pc(0xB4F1D5), 0x84)

    # # Skip map screens when starting after game over
    # rom.write_u16(snes2pc(0x81911F), 0x0006)

    escape_spoiler = update_escape_timer(rom, map, sm_json_data, difficulty)

    spoiler_data = {
        'summary': randomizer.spoiler_summary,
        'route': randomizer.spoiler_route,
        'escape': escape_spoiler,
        # 'items': spoiler_items,
    }

    memory_file = BytesIO()
    files = [
        (output_file_prefix + '.sfc', rom.bytes_io.getvalue()),
        (output_file_prefix + '-config.json', json.dumps(config, indent=2)),
        (output_file_prefix + '-spoiler.json', json.dumps(spoiler_data, indent=2)),
        (output_file_prefix + '-map.png', map_png_bytes),
        (output_file_prefix + '-map-vanilla.png', map_orig_png_bytes),
    ]
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for file_name, file_data in files:
            data = zipfile.ZipInfo(file_name, date_time=datetime.datetime.utcnow().utctimetuple()[:6])
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, file_data)
    memory_file.seek(0)
    return flask.send_file(memory_file, download_name=output_file_prefix + '.zip')

    # return flask.send_file(io.BytesIO(rom.byte_buf), mimetype='application/octet-stream', download_name=output_filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
