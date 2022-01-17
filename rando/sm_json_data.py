import abc
from typing import Set, Dict
import json
import logging
import pathlib

class DifficultyConfig:
    tech: Set[str]  # Set of names of enabled tech (https://github.com/miketrethewey/sm-json-data/blob/master/tech.json)
    shine_charge_tiles: int  # Minimum number of tiles required to shinespark

class GameState:
    difficulty: DifficultyConfig
    items: Set[str]   # Set of collected items
    flags: Set[str]   # Set of activated flags

class Condition:
    @abc.abstractmethod
    def is_accessible(self, state: GameState) -> bool:
        raise NotImplemented

def get_plm_type_item_index(plm_type):
    assert 0xEED7 <= plm_type <= 0xEFCF
    assert plm_type % 4 == 3
    i = ((plm_type - 0xEED7) // 4) % 21
    return i

class ConstantCondition:
    def __init__(self, cond: bool):
        self.cond = cond

    def is_accessible(self, state: GameState) -> bool:
        return self.cond

class TechCondition:
    def __init__(self, tech: str):
        self.tech = tech

    def is_accessible(self, state: GameState) -> bool:
        return self.tech in state.difficulty.tech


class ShineChargeCondition:
    def __init__(self, tiles: int):
        self.tiles = tiles

    def is_accessible(self, state: GameState) -> bool:
        return self.tiles <= state.difficulty.shine_charge_tiles

class ItemCondition:
    def __init__(self, item: str):
        self.item = item

    def is_accessible(self, state: GameState) -> bool:
        return self.item in state.items


class FlagCondition:
    def __init__(self, flag: str):
        self.flag = flag

    def is_accessible(self, state: GameState) -> bool:
        return self.flag in state.flags


class AndCondition:
    def __init__(self, conditions):
        self.conditions = conditions

    def is_accessible(self, state: GameState) -> bool:
        return all(cond.is_accessible(state) for cond in self.conditions)

class OrCondition:
    def __init__(self, conditions):
        self.conditions = conditions

    def is_accessible(self, state: GameState) -> bool:
        return any(cond.is_accessible(state) for cond in self.conditions)


class SMJsonData:
    def __init__(self, path_to_sm_json_data):
        tech_json = json.load(open(f'{path_to_sm_json_data}/tech.json', 'r'))
        self.tech_name_set = set([tech['name'] for tech in tech_json['techs']])

        items_json = json.load(open(f'{path_to_sm_json_data}/items.json', 'r'))
        item_categories = ['implicitItems', 'upgradeItems', 'expansionItems']
        self.item_set = set(x if isinstance(x, str) else x['name'] for c in item_categories for x in items_json[c])
        self.flags_set = set(items_json['gameFlags'])

        helpers_json = json.load(open(f'{path_to_sm_json_data}/helpers.json', 'r'))
        self.helpers = {}
        for helper_json in helpers_json['helpers']:
            cond = self.make_condition(helper_json['requires'])
            self.helpers[helper_json['name']] = cond

    def make_condition(self, json_data):
        if isinstance(json_data, str):
            if json_data in self.tech_name_set:
                return TechCondition(json_data)
            if json_data in self.item_set:
                return ItemCondition(json_data)
            if json_data in self.flags_set:
                return FlagCondition(json_data)
            if json_data in self.helpers.keys():
                return self.helpers[json_data]
        elif isinstance(json_data, list):
            return AndCondition([self.make_condition(x) for x in json_data])
        elif isinstance(json_data, dict):
            assert len(json_data) == 1
            key = next(iter(json_data.keys()))
            val = json_data[key]
            if key == 'or':
                return OrCondition([self.make_condition(x) for x in val])
            if key == 'and':
                return AndCondition([self.make_condition(x) for x in val])
            if key == 'ammo':
                # For now we ignore ammo quantity, just require one of the ammo type
                item_type = val['type']
                assert item_type in self.item_set
                return ItemCondition(item_type)
            if key == 'canShineCharge':
                return ShineChargeCondition(val['usedTiles'])
            if key == 'heatFrames':
                # For now we keep canHeatRun=False, so heat frames are irrelevant.
                return ConstantCondition(True)
            if key in ('lavaFrames', 'lavaPhysicsFrames', 'acidFrames', 'enemyDamage', 'spikeHits', 'hibashiHits', 'energyAtMost'):
                # For now we ignore energy requirements.
                return ConstantCondition(True)
            if key in ('enemyKill', 'resetRoom', 'canComeInCharged', 'adjacentRunway', 'previousStratProperty', 'previousNode'):
                # Ignore these for now.
                return ConstantCondition(False)

        raise RuntimeError("Unrecognized condition: {}".format(json_data))

    def process_region(self, json_data):
        for room_json in json_data['rooms']:
            for node_json in room_json['nodes']:
                pass
            for link_json in room_json['links']:
                for link_to_json in link_json['to']:
                    strats = []
                    for strat_json in link_to_json['strats']:
                        strats.append(self.make_condition(strat_json['requires']))
                    cond = OrCondition(strats)


sm_json_data_path = "sm-json-data/"
sm_json_data = SMJsonData(sm_json_data_path)

region_files = [str(f) for f in pathlib.Path(f"{sm_json_data_path}/region").glob("**/*.json")]
for filename in region_files:
    # filename = region_files[2]
    logging.info("Processing {}".format(filename))
    region_data = json.load(open(filename, 'r'))
    sm_json_data.process_region(region_data)