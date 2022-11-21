import abc
import copy
from typing import Set, Dict, List
import json
import logging
import pathlib
import dataclasses
import numpy as np
from dataclasses import dataclass

# C++ module for computing reachability in graph
# (TODO: set this up more properly)
import reachability

# An upper bound on the possible max amount of resources
ENERGY_LIMIT = 1899
MISSILE_LIMIT = 230
SUPER_MISSILE_LIMIT = 50
POWER_BOMB_LIMIT = 50


@dataclass
class DifficultyConfig:
    tech: Set[str]  # Set of names of enabled tech (https://github.com/miketrethewey/sm-json-data/blob/master/tech.json)
    shine_charge_tiles: int  # Minimum number of tiles required to shinespark
    multiplier: float  # Multiplier for energy/ammo requirements (1.0 is highest difficulty, larger values are easier)


@dataclass
class Consumption:
    possible: bool = True
    energy: int = 0
    missiles: int = 0
    super_missiles: int = 0
    power_bombs: int = 0


@dataclass
class GameState:
    difficulty: DifficultyConfig
    items: Set[str]  # Set of collected items
    flags: Set[str]  # Set of activated flags
    weapons: Set[str]  # Set of non-situational weapons (derived from collected items)
    num_energy_tanks: int
    num_reserves: int
    max_energy: int  # (derived from num_energy_tanks and num_reserves)
    max_missiles: int
    max_super_missiles: int
    max_power_bombs: int
    current_energy: int
    current_missiles: int
    current_super_missiles: int
    current_power_bombs: int
    vertex_index: int  # Current node (representing room and location within room)


class Condition:
    @abc.abstractmethod
    def get_consumption(self, state: GameState) -> Consumption:
        raise NotImplementedError


class Target:
    @abc.abstractmethod
    def collect(self, state: GameState) -> GameState:
        raise NotImplementedError


class FlagTarget(Target):
    def __init__(self, flag_name: str):
        self.flag_name = flag_name

    def collect(self, state: GameState) -> GameState:
        new_state = copy.deepcopy(state)
        new_state.flags.add(self.flag_name)
        return new_state


class ItemTarget(Target):
    def __init__(self, item_name: str):
        self.item_name = item_name

    def collect(self, state: GameState) -> GameState:
        new_state = copy.deepcopy(state)
        new_state.items.add(self.item_name)
        if self.item_name == 'Missile':
            new_state.max_missiles += 5
            new_state.current_missiles += 5
        elif self.item_name == 'Super':
            new_state.max_super_missiles += 5
            new_state.current_super_missiles += 5
        elif self.item_name == 'PowerBomb':
            new_state.max_power_bombs += 5
            new_state.current_power_bombs += 5
        elif self.item_name == 'ETank':
            new_state.num_energy_tanks += 1
            new_state.max_energy += 100
            new_state.current_energy = new_state.max_energy
        elif self.item_name == 'ReserveTank':
            new_state.num_reserves += 1
            new_state.max_energy += 100
        return new_state

# def get_plm_type_item_index(plm_type):
#     assert 0xEED7 <= plm_type <= 0xEFCF
#     assert plm_type % 4 == 3
#     i = ((plm_type - 0xEED7) // 4) % 21
#     return i

zero_consumption = Consumption()
impossible_consumption = Consumption(possible=False)


class FreeCondition(Condition):
    def get_consumption(self, state: GameState) -> Consumption:
        return zero_consumption


class ImpossibleCondition(Condition):
    def get_consumption(self, state: GameState) -> Consumption:
        return impossible_consumption


class TechCondition(Condition):
    def __init__(self, tech: str):
        self.tech = tech

    def get_consumption(self, state: GameState) -> Consumption:
        if self.tech in state.difficulty.tech:
            return zero_consumption
        else:
            return impossible_consumption

    def __repr__(self):
        return "Tech(" + self.tech + ")"


class ShineChargeCondition(Condition):
    def __init__(self, tiles: int, frames: int):
        self.tiles = tiles
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if "SpeedBooster" in state.items and self.tiles >= state.difficulty.shine_charge_tiles:
            return Consumption(energy=self.frames)
        else:
            return impossible_consumption


class ItemCondition(Condition):
    def __init__(self, item: str):
        self.item = item

    def get_consumption(self, state: GameState) -> Consumption:
        if self.item in state.items:
            return zero_consumption
        else:
            return impossible_consumption

    def __repr__(self):
        return "Item(" + self.item + ")"


class FlagCondition(Condition):
    def __init__(self, flag: str):
        self.flag = flag

    def get_consumption(self, state: GameState) -> Consumption:
        if self.flag in state.flags:
            return zero_consumption
        else:
            return impossible_consumption

    def __repr__(self):
        return "Flag(" + self.flag + ")"


class MissileCondition(Condition):
    def __init__(self, amount: int):
        self.amount = amount

    def get_consumption(self, state: GameState) -> Consumption:
        return Consumption(missiles=self.amount)

    def __repr__(self):
        return "Missile({})".format(self.amount)


class SuperMissileCondition(Condition):
    def __init__(self, amount: int):
        self.amount = amount

    def get_consumption(self, state: GameState) -> Consumption:
        return Consumption(super_missiles=self.amount)

    def __repr__(self):
        return "SuperMissile({})".format(self.amount)


class PowerBombCondition(Condition):
    def __init__(self, amount: int):
        self.amount = amount

    def get_consumption(self, state: GameState) -> Consumption:
        return Consumption(power_bombs=self.amount)

    def __repr__(self):
        return "PowerBomb({})".format(self.amount)


class EnergyCondition(Condition):
    def __init__(self, amount: int):
        self.amount = amount

    def get_consumption(self, state: GameState) -> Consumption:
        return Consumption(energy=self.amount)

    def __repr__(self):
        return "Energy({})".format(self.amount)


# def max_consumption(a: Consumption, b: Consumption) -> Consumption:
#     return Consumption(energy=max(a.energy, b.energy),
#                        missiles=max(a.missiles, b.missiles),
#                        super_missiles=max(a.super_missiles, b.super_missiles),
#                        power_bombs=max(a.power_bombs, b.power_bombs))

def sum_consumption(a: Consumption, b: Consumption) -> Consumption:
    if a.possible and b.possible:
        return Consumption(energy=a.energy + b.energy,
                           missiles=a.missiles + b.missiles,
                           super_missiles=a.super_missiles + b.super_missiles,
                           power_bombs=a.power_bombs + b.power_bombs)
    else:
        return Consumption(possible=False)


# def get_consumption_scalar_cost(c: Consumption, state: GameState) -> float:
#     eps = 1e-5
#     energy_cost = c.energy / (state.max_energy + eps)
#     missile_cost = c.missiles / (state.max_missiles + eps)
#     super_cost = c.super_missiles / (state.max_super_missiles + eps)
#     pb_cost = c.power_bombs / (state.max_power_bombs + eps)
#     return energy_cost + missile_cost + super_cost + pb_cost
#
# def min_consumption(a: Consumption, b: Consumption, state: GameState) -> Consumption:
#     scalar_cost_a = get_consumption_scalar_cost(a, state)
#     scalar_cost_b = get_consumption_scalar_cost(b, state)
#     if scalar_cost_a <= scalar_cost_b:
#         return a
#     else:
#         return b

def min_consumption(a: Consumption, b: Consumption, state: GameState) -> Consumption:
    a_tuple = (not a.possible, a.energy, a.power_bombs, a.super_missiles, a.missiles)
    b_tuple = (not b.possible, b.energy, b.power_bombs, b.super_missiles, b.missiles)
    if a_tuple <= b_tuple:
        return a
    else:
        return b


class AndCondition(Condition):
    def __init__(self, conditions):
        self.conditions = conditions

    def get_consumption(self, state: GameState) -> Consumption:
        consumption = zero_consumption
        for cond in self.conditions:
            consumption = sum_consumption(consumption, cond.get_consumption(state))
        return consumption

    def __repr__(self):
        return "And(" + ','.join(str(c) for c in self.conditions) + ")"


class OrCondition(Condition):
    def __init__(self, conditions):
        self.conditions = conditions

    def get_consumption(self, state: GameState) -> Consumption:
        consumption = impossible_consumption
        for cond in self.conditions:
            consumption = min_consumption(consumption, cond.get_consumption(state), state)
        return consumption

    def __repr__(self):
        return "Or(" + ','.join(str(c) for c in self.conditions) + ")"


class HeatCondition(Condition):
    def __init__(self, frames):
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items:
            return zero_consumption
        elif 'Gravity' in state.items:
            return Consumption(energy=(self.frames + 7) // 8)
        else:
            return Consumption(energy=(self.frames + 3) // 4)

    def __repr__(self):
        return "Heat({})".format(self.frames)


class LavaCondition(Condition):
    def __init__(self, frames):
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return zero_consumption
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=(self.frames + 3) // 4)
        else:
            return Consumption(energy=(self.frames + 1) // 2)

    def __repr__(self):
        return "Lava({})".format(self.frames)


class LavaPhysicsCondition(Condition):
    def __init__(self, frames):
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items:
            return Consumption(energy=(self.frames + 3) // 4)
        else:
            return Consumption(energy=(self.frames + 1) // 2)

    def __repr__(self):
        return "LavaPhysics({})".format(self.frames)


class AcidCondition(Condition):
    def __init__(self, frames):
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=(3 * self.frames + 7) // 8)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=(3 * self.frames + 3) // 4)
        else:
            return Consumption(energy=(3 * self.frames + 1) // 2)

    def __repr__(self):
        return "Acid({})".format(self.frames)


class SpikeHitCondition(Condition):
    def __init__(self, hits):
        self.hits = hits

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=15 * self.hits)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=30 * self.hits)
        else:
            return Consumption(energy=60 * self.hits)

    def __repr__(self):
        return "SpikeHit({})".format(self.hits)


class ThornHitCondition(Condition):
    def __init__(self, hits):
        self.hits = hits

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=4 * self.hits)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=8 * self.hits)
        else:
            return Consumption(energy=16 * self.hits)

    def __repr__(self):
        return "ThornHit({})".format(self.hits)


class HibashiHitCondition(Condition):
    def __init__(self, hits):
        self.hits = hits

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=7 * self.hits)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=15 * self.hits)
        else:
            return Consumption(energy=30 * self.hits)

    def __repr__(self):
        return "HibashiHit({})".format(self.hits)


class DraygonElectricityCondition(Condition):
    def __init__(self, frames):
        self.frames = frames

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=self.frames // 4)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=self.frames // 2)
        else:
            return Consumption(energy=self.frames)

    def __repr__(self):
        return "DraygonElectricity({})".format(self.frames)


class EnemyDamageCondition(Condition):
    def __init__(self, base_damage):
        self.base_damage = base_damage

    def get_consumption(self, state: GameState) -> Consumption:
        if 'Varia' in state.items and 'Gravity' in state.items:
            return Consumption(energy=self.base_damage // 4)
        elif 'Varia' in state.items or 'Gravity' in state.items:
            return Consumption(energy=self.base_damage // 2)
        else:
            return Consumption(energy=self.base_damage)

    def __repr__(self):
        return "EnemyDamage({})".format(self.frames)


class EnemyKillCondition(Condition):
    def __init__(self, vulnerable_weapons):
        self.vulnerable_weapons = vulnerable_weapons

    def get_consumption(self, state: GameState) -> Consumption:
        if state.weapons.isdisjoint(self.vulnerable_weapons):
            return impossible_consumption
        else:
            return zero_consumption

    def __repr__(self):
        return "EnemyKill({})".format(self.vulnerable_weapons)


class RefillCondition(Condition):
    def __init__(self, drops_energy, drops_missile, drops_supers, drops_pbs):
        self.consumption = Consumption(
            energy=-ENERGY_LIMIT if drops_energy else 0,
            missiles=-MISSILE_LIMIT if drops_missile else 0,
            super_missiles=-SUPER_MISSILE_LIMIT if drops_supers else 0,
            power_bombs=-POWER_BOMB_LIMIT if drops_pbs else 0)

    def get_consumption(self, state: GameState) -> Consumption:
        return self.consumption

    def __repr__(self):
        return "Refill({})".format(self.consumption)


# Helper function to simplify AndCondition in case of 0 or 1 conditions
def make_and_condition(conditions: List[Condition]):
    out_conditions = []
    for cond in conditions:
        if isinstance(cond, FreeCondition):
            pass
        elif isinstance(cond, ImpossibleCondition):
            return ImpossibleCondition()
        elif isinstance(cond, AndCondition):
            for c in cond.conditions:
                out_conditions.append(c)
        else:
            out_conditions.append(cond)
    if len(out_conditions) == 0:
        return FreeCondition()
    if len(out_conditions) == 1:
        return out_conditions[0]
    else:
        return AndCondition(out_conditions)


# Helper function to simplify OrCondition in specific cases
def make_or_condition(conditions: List[Condition]):
    out_conditions = []
    for cond in conditions:
        if isinstance(cond, FreeCondition):
            return FreeCondition()
        elif isinstance(cond, ImpossibleCondition):
            pass
        elif isinstance(cond, OrCondition):
            for c in cond.conditions:
                out_conditions.append(c)
        else:
            out_conditions.append(cond)
    if len(out_conditions) == 0:
        return ImpossibleCondition()
    elif len(out_conditions) == 1:
        return out_conditions[0]
    else:
        return OrCondition(out_conditions)


@dataclass
class Link:
    from_index: int  # index in SMJsonData.node_list
    to_index: int  # index in SMJsonData.node_list
    cond: Condition
    strat_name: str


def has_key(x, k):
    if isinstance(x, list):
        return any(has_key(y, k) for y in x)
    elif isinstance(x, dict):
        if k in x.keys():
            return True
        return any(has_key(y, k) for y in x.values())
    else:
        return False


def find_key(x, key, prefix):
    if isinstance(x, list):
        return [z for i, y in enumerate(x) for z in find_key(y, key, prefix + '.{}'.format(i))]
    elif isinstance(x, dict):
        if key in x.keys():
            out = [prefix + '.{}'.format(key)]
        else:
            out = []
        for k, v in x.items():
            out += find_key(v, key, prefix + '.{}'.format(k))
        return out
    else:
        return []


class SMJsonData:
    def __init__(self, sm_json_data_path):
        items_json = json.load(open(f'{sm_json_data_path}/items.json', 'r'))
        item_categories = ['implicitItems', 'upgradeItems', 'expansionItems']
        self.item_set = set(x if isinstance(x, str) else x['name'] for c in item_categories for x in items_json[c])
        self.flags_set = set(items_json['gameFlags'])
        self.helpers = {}

        tech_json = json.load(open(f'{sm_json_data_path}/tech.json', 'r'))
        self.tech_json_dict = {tech['name']: tech for tech in tech_json['techs']}
        self.tech_name_set = set(self.tech_json_dict.keys())

        helpers_json = json.load(open(f'{sm_json_data_path}/helpers.json', 'r'))
        self.helpers_json_dict = {helper['name']: helper for helper in helpers_json['helpers']}

        enemies_json = json.load(open(f'{sm_json_data_path}/enemies/main.json', 'r'))
        bosses_json = json.load(open(f'{sm_json_data_path}/enemies/bosses/main.json', 'r'))
        self.enemies_json_dict = {enemy['name']: enemy for enemy in enemies_json['enemies'] + bosses_json['enemies']}

        weapons_json = json.load(open(f'{sm_json_data_path}/weapons/main.json', 'r'))
        self.weapons_json_dict = {weapon['name']: weapon for weapon in weapons_json['weapons']}
        self.considered_weapons_set = set(
            [weapon['name'] for weapon in self.weapons_json_dict.values() if self.is_weapon_considered(weapon)])

        self.enemy_vulnerability_dict = {enemy['name']: self.get_enemy_vulnerabilities(enemy)
                                         for enemy in self.enemies_json_dict.values()}

        self.cond_dict = {}

        for tech_name in self.tech_json_dict.keys():
            self.register_tech_condition(tech_name)

        for helper_name in self.helpers_json_dict.keys():
            self.register_helper_condition(helper_name)
        # TODO: Patch h_heatProof to only use Varia, not Gravity
        # TODO: Check enemy-count grey door locks to make sure they all unlock f_zebesAwake
        # TODO: Patch Statues room to open it up
        # TODO: Patch out backdoor Shaktool?

        self.vertex_list = []  # List of triples (room_id, node_id, obstacle_bitmask) in order
        self.vertex_index_dict = {}  # Maps (room_id, node_id, obstacle_bitmask) to integer, the index in self.vertex_index_list
        self.num_obstacles_dict = {}  # Maps room_id to number of obstacles in room
        self.node_ptr_dict = {}  # Maps (room_id, node_id) to node pointer
        self.target_dict = []  # Dict mapping pair (room_id, node_id) to either a node pointer (for an item location) or flag name (for a node yielding a flag)
        self.link_list = []
        self.region_json_dict = {}
        self.target_dict = {}  # Maps vertex_id to Target

        region_files = [str(f) for f in pathlib.Path(f"{sm_json_data_path}/region").glob("**/*.json")]
        for filename in region_files:
            # logging.info("Processing {}".format(filename))
            if "ceres" not in filename:
                region_data = json.load(open(filename, 'r'))
                region_data = self.preprocess_region(region_data)
                self.region_json_dict[filename] = region_data
                self.process_region(region_data)
        # Add Pants Room in-room transition
        from_index = self.vertex_index_dict[(220, 2, 0)]  # Pants Room
        to_index = self.vertex_index_dict[(322, 1, 0)]  # East Pants Room
        self.link_list.append(Link(from_index, to_index, FreeCondition(), "Pants Room in-room transition"))

        self.door_ptr_pair_dict = {}
        connection_files = [str(f) for f in pathlib.Path(f"{sm_json_data_path}/connection").glob("**/*.json")]
        for filename in connection_files:
            if "ceres" not in filename:
                connection_data = json.load(open(filename, 'r'))
                self.process_connections(connection_data)

    def is_weapon_considered(self, weapon_json: dict) -> bool:
        if weapon_json['situational']:
            return False
        if 'shotRequires' in weapon_json:
            return False
        return True

    def get_weapons(self, items: Set[str]) -> Set[str]:
        weapons_list = []
        for weapon in self.weapons_json_dict.values():
            if not self.is_weapon_considered(weapon):
                continue
            if set(weapon['useRequires']).issubset(items):
                weapons_list.append(weapon['name'])
        return set(weapons_list)

    def get_enemy_vulnerabilities(self, enemy_json: dict) -> Set[str]:
        invul = set(enemy_json['invul'])
        vul_list = []
        for weapon in self.weapons_json_dict.values():
            if not self.is_weapon_considered(weapon):
                continue
            if any(cat in invul for cat in weapon['categories'] + [weapon['name']]):
                continue
            vul_list.append(weapon['name'])
        return set(vul_list)

    def register_tech_condition(self, name):
        if name in self.cond_dict:
            if self.cond_dict[name] is None:
                raise RuntimeError(f"Circular dependency in {name}")
        self.cond_dict[name] = None  # Set a sentinel value for detecting potential circular dependencies
        conds = [self.make_condition(c) for c in self.tech_json_dict[name]['requires']]
        self.cond_dict[name] = make_and_condition([TechCondition(name), *conds])

    def register_helper_condition(self, name):
        if name in self.cond_dict:
            if self.cond_dict[name] is None:
                raise RuntimeError(f"Circular dependency in {name}")
        self.cond_dict[name] = None  # Set a sentinel value for detecting potential circular dependencies
        self.cond_dict[name] = self.make_condition(self.helpers_json_dict[name]['requires'])

    def make_condition(self, json_data):
        if isinstance(json_data, str):
            if json_data == 'never':
                return ImpossibleCondition()
            if json_data in self.item_set:
                return ItemCondition(json_data)
            if json_data in self.flags_set:
                return FlagCondition(json_data)
            if json_data in self.cond_dict.keys():
                return self.cond_dict[json_data]
            if json_data in self.tech_json_dict.keys():
                self.register_tech_condition(json_data)
                return self.cond_dict[json_data]
            if json_data in self.helpers_json_dict.keys():
                self.register_helper_condition(json_data)
                return self.cond_dict[json_data]
        elif isinstance(json_data, list):
            return make_and_condition([self.make_condition(x) for x in json_data])
        elif isinstance(json_data, dict):
            assert len(json_data) == 1
            key = next(iter(json_data.keys()))
            val = json_data[key]
            if key == 'or':
                return make_or_condition([self.make_condition(x) for x in val])
            if key == 'and':
                return make_and_condition([self.make_condition(x) for x in val])
            if key == 'ammo':
                item_type = val['type']
                if item_type == 'Missile':
                    return MissileCondition(val['count'])
                elif item_type == 'Super':
                    return SuperMissileCondition(val['count'])
                elif item_type == 'PowerBomb':
                    return PowerBombCondition(val['count'])
                else:
                    raise NotImplementedError("Unexpected 'ammo' type: {}".format(item_type))
            if key == 'ammoDrain':
                # This only occurs in the Mother Brain fight. We treat it as free because it would be a pain (and
                # inefficient) to try to incorporate this as a new kind of edge in our graph traversal. This should
                # still be correct if we assume that ammo is not needed in the escape.
                # TODO: Make sure this assumption is correct.
                # (Alternatively, patch the fight to skip draining the ammo.)
                return FreeCondition()
            if key == 'canShineCharge':
                return ShineChargeCondition(val['usedTiles'], val['shinesparkFrames'])
            if key == 'heatFrames':
                return HeatCondition(val)
            if key == 'lavaFrames':
                return LavaCondition(val)
            if key == 'lavaPhysicsFrames':
                return LavaPhysicsCondition(val)
            if key == 'acidFrames':
                return AcidCondition(val)
            if key == 'draygonElectricityFrames':
                return DraygonElectricityCondition(val)
            if key == 'spikeHits':
                return SpikeHitCondition(val)
            if key == 'thornHits':
                return ThornHitCondition(val)
            if key == 'hibashiHits':
                return HibashiHitCondition(val)
            if key == 'enemyDamage':
                if val['enemy'] not in self.enemies_json_dict.keys():
                    raise NotImplementedError('In enemyDamage, unexpected enemy: {}'.format(val['enemy']))
                enemy_dict = self.enemies_json_dict[val['enemy']]
                attacks = {attack['name']: attack['baseDamage'] for attack in enemy_dict['attacks']}
                if val['type'] not in attacks.keys():
                    raise NotImplementedError(
                        'In enemyDamage for {}, unexpected enemy attack: {}'.format(val['enemy'], val['type']))
                return EnemyDamageCondition(val['hits'] * attacks[val['type']])
            if key == 'energyAtMost':
                # This is only used for the Baby Metroid drain down to 1 energy (and for the Ceres Ridley fight, which
                # is irrelevant for us). We treat it as free because it would be a pain (and inefficient) to try to
                # incorporate this as a new kind of edge in our graph traversal. If canBabyMetroidAvoid tech is
                # enabled, then the drain can be skipped, making this indeed free; otherwise we need to figure
                # out some other way to make this correct.
                # TODO: If canBabyMetroidAvoid is not enabled, patch the game to skip the baby cutscene?
                return FreeCondition()
            if key == 'enemyKill':
                # We only consider enemy kill methods that are non-situational and do not require ammo.
                # TODO: Consider all methods.
                conds = []
                enemy_set = set()
                for enemy_group in val['enemies']:
                    for enemy in enemy_group:
                        enemy_set.add(enemy)
                if 'explicitWeapons' in val:
                    allowed_weapons = set(val['explicitWeapons'])
                else:
                    allowed_weapons = self.considered_weapons_set
                if 'excludedWeapons' in val:
                    allowed_weapons = allowed_weapons.difference(set(val['excludedWeapons']))
                for enemy in enemy_set:
                    conds.append(
                        EnemyKillCondition(allowed_weapons.intersection(self.enemy_vulnerability_dict[enemy])))
                return make_and_condition(conds)
            if key == 'previousNode':
                # Currently this is used only in the Early Supers quick crumble and Mission Impossible strats and is
                # redundant in both cases, so we treat it as free.
                return FreeCondition()
            if key == 'resetRoom':
                # In all the places where this is required (excluding runways and canComeInCharged which we are not
                # yet taking into account), it seems to be essentially unnecessary (ignoring the
                # possibility of needing to take a small amount of heat damage in an adjacent room to exit and
                # reenter), so for now we treat it as free.
                return FreeCondition()
            if key == 'previousStratProperty':
                # This is only used in one place in Crumble Shaft, where it doesn't seem to be necessary.
                return FreeCondition()
            if key in ('canComeInCharged', 'adjacentRunway'):
                # For now assume we can't do these.
                return ImpossibleCondition()
            # TODO:
            # - Boss flags
            # - Zebes awake flag

        raise RuntimeError("Unrecognized condition: {}".format(json_data))

    def preprocess_room(self, raw_room_json):
        room_json = copy.deepcopy(raw_room_json)
        next_node_id = max(node_json['id'] for node_json in room_json['nodes']) + 1
        extra_node_list = []
        extra_link_list = []
        for node_json in room_json['nodes']:
            if 'locks' in node_json and node_json['nodeType'] not in ('door', 'entrance'):
                assert len(node_json['locks']) == 1
                base_node_name = node_json['name']
                lock = node_json['locks'][0]
                yields = node_json.get('yields')
                del node_json['locks']
                new_node_json = copy.deepcopy(node_json)
                if yields is not None:
                    del node_json['yields']
                node_json['name'] = base_node_name + ' (locked)'
                node_json['nodeType'] = 'junction'

                new_node_json['id'] = next_node_id
                new_node_json['name'] = base_node_name + ' (unlocked)'
                if yields is not None:
                    new_node_json['yields'] = yields
                extra_node_list.append(new_node_json)

                link_to = {
                    'from': node_json['id'],
                    'to': [{
                        'id': next_node_id,
                        'strats': lock['unlockStrats']
                    }]
                }
                link_from = {
                    'from': next_node_id,
                    'to': [{
                        'id': node_json['id'],
                        'strats': [{
                            'name': 'Base',
                            'notable': False,
                            'requires': [],
                        }],
                    }]
                }
                extra_link_list.append(link_to)
                extra_link_list.append(link_from)

                next_node_id += 1

        room_json['nodes'] += extra_node_list
        room_json['links'] += extra_link_list
        return room_json

    def preprocess_region(self, raw_json_data):
        json_data = copy.deepcopy(raw_json_data)
        json_data['rooms'] = [self.preprocess_room(room) for room in raw_json_data['rooms']]
        return json_data

    def process_region(self, json_data):
        for room_json in json_data['rooms']:
            room_id = room_json['id']
            if 'obstacles' in room_json:
                obstacles_dict = {obstacle['id']: i for i, obstacle in enumerate(room_json['obstacles'])}
            else:
                obstacles_dict = {}
            self.num_obstacles_dict[room_id] = len(obstacles_dict)
            # if has_key(room_json, 'previousStratProperty'):
            #     print(f"room='{room_json['name']}'")
            #     for s in find_key(room_json, 'previousStratProperty', ''):
            #         print(s)
            for node_json in room_json['nodes']:
                pair = (room_id, node_json['id'])
                if 'nodeAddress' in node_json:
                    node_ptr = int(node_json['nodeAddress'], 16)
                    # Convert East Pants Room door pointers to corresponding Pants Room pointers
                    if node_ptr == 0x1A7BC:
                        node_ptr = 0x1A798
                    if node_ptr == 0x1A7B0:
                        node_ptr = 0x1A7A4
                else:
                    node_ptr = None
                self.node_ptr_dict[pair] = node_ptr
                if node_json['nodeType'] == 'item':
                    self.target_dict[pair] = node_ptr
                elif 'yields' in node_json:
                    self.target_dict[pair] = node_json['yields'][0]
                for obstacle_bitmask in range(2 ** len(obstacles_dict)):
                    triple = (room_id, node_json['id'], obstacle_bitmask)
                    # print("added:", triple)
                    self.vertex_index_dict[triple] = len(self.vertex_list)
                    self.vertex_list.append(triple)
            for node_json in room_json['nodes']:
                # if 'locks' in node_json and node_json['nodeType'] != 'door':
                #     print(f"room='{room_json['name']}', node='{node_json['name']}', nodeType='{node_json.get('nodeType')}', yields='{node_json.get('yields')}'")
                # if 'yields' in node_json:
                #     print(f"room='{room_json['name']}', node='{node_json['id']}', yields='{node_json.get('yields')}'")
                # if 'yields' in node_json:
                #     self.yields_pair_list.append((room_id, node_json['id']))
                # TODO: handle spawnAt more correctly.
                if 'spawnAt' in node_json:
                    from_index = self.vertex_index_dict[(room_id, node_json['id'], 0)]
                    to_index = self.vertex_index_dict[(room_id, node_json['spawnAt'], 0)]
                    self.link_list.append(Link(from_index, to_index, FreeCondition(), "spawnAt"))
                if 'utility' in node_json:
                    for obstacle_bitmask in range(2 ** len(obstacles_dict)):
                        triple = (room_id, node_json['id'], obstacle_bitmask)
                        index = self.vertex_index_dict[triple]
                        fills_energy = 'energy' in node_json['utility']
                        fills_missiles = 'missile' in node_json['utility']
                        fills_supers = 'super' in node_json['utility']
                        fills_pbs = 'powerbomb' in node_json['utility']
                        cond = RefillCondition(fills_energy, fills_missiles, fills_supers, fills_pbs)
                        self.link_list.append(Link(index, index, cond, "Refill"))
            for enemy in (room_json['enemies'] if 'enemies' in room_json else []):
                if 'farmCycles' in enemy:
                    # We're ignoring "requires" here. TOOD: Fix this if it is a problem.
                    enemy_json = self.enemies_json_dict[enemy['enemyName']]
                    drops = enemy_json['drops']
                    drops_pbs = drops['powerBomb'] > 0
                    drops_supers = drops['super'] > 0
                    drops_missile = drops_pbs | drops_supers | (drops['missile'] > 0)
                    drops_energy = drops_pbs | drops_supers | (drops['bigEnergy'] > 0) | (drops['smallEnergy'] > 0)
                    farm_name = "Farm {}".format(enemy['enemyName'])
                    for node_id in enemy['homeNodes']:
                        for obstacle_bitmask in range(2 ** len(obstacles_dict)):
                            index = self.vertex_index_dict[(room_id, node_id, obstacle_bitmask)]
                            cond = RefillCondition(drops_energy, drops_missile, drops_supers, drops_pbs)
                            self.link_list.append(Link(index, index, cond, farm_name))
            for link_json in room_json['links']:
                for link_to_json in link_json['to']:
                    for strat_json in link_to_json['strats']:
                        for from_obstacle_bitmask in range(2 ** len(obstacles_dict)):
                            requires = strat_json['requires']
                            to_obstacle_bitmask = from_obstacle_bitmask
                            if "obstacles" in strat_json:
                                for obstacle in strat_json['obstacles']:
                                    obstacle_idx = obstacles_dict[obstacle['id']]
                                    to_obstacle_bitmask |= 1 << obstacle_idx
                                    if (1 << obstacle_idx) & from_obstacle_bitmask == 0:
                                        requires = requires + obstacle['requires']
                                        if 'requires' in room_json['obstacles'][obstacle_idx]:
                                            requires = requires + room_json['obstacles'][obstacle_idx]['requires']
                                    if "additionalObstacles" in obstacle:
                                        for additional_obstacle_id in obstacle['additionalObstacles']:
                                            additional_obstacle_idx = obstacles_dict[additional_obstacle_id]
                                            to_obstacle_bitmask |= 1 << additional_obstacle_idx
                            cond = self.make_condition(requires)
                            from_id = link_json['from']
                            from_index = self.vertex_index_dict[(room_id, from_id, from_obstacle_bitmask)]
                            to_id = link_to_json['id']
                            to_index = self.vertex_index_dict[(room_id, to_id, to_obstacle_bitmask)]
                            # if not isinstance(cond, ImpossibleCondition):
                            self.link_list.append(Link(from_index, to_index, cond, strat_json['name']))

    def process_connections(self, json_data):
        for connection in json_data['connections']:
            assert len(connection['nodes']) == 2
            src_pair = (connection['nodes'][0]['roomid'], connection['nodes'][0]['nodeid'])
            dst_pair = (connection['nodes'][1]['roomid'], connection['nodes'][1]['nodeid'])
            src_ptr = self.node_ptr_dict.get(src_pair)
            dst_ptr = self.node_ptr_dict.get(dst_pair)
            if src_ptr is not None or dst_ptr is not None:
                self.door_ptr_pair_dict[(src_ptr, dst_ptr)] = src_pair
                self.door_ptr_pair_dict[(dst_ptr, src_ptr)] = dst_pair

    def get_graph(self, state: GameState, door_edges) -> np.array:
        graph = np.zeros([len(self.link_list) + len(door_edges), 6], dtype=np.int16)
        i = 0
        for link in self.link_list:
            consumption = link.cond.get_consumption(state)
            if consumption.possible:
                graph[i, 0] = link.from_index
                graph[i, 1] = link.to_index
                graph[i, 2] = consumption.energy
                graph[i, 3] = consumption.missiles
                graph[i, 4] = consumption.super_missiles
                graph[i, 5] = consumption.power_bombs
                i += 1
        for (src_index, dst_index) in door_edges:
            graph[i, 0] = src_index
            graph[i, 1] = dst_index
            graph[i, 2:6] = 0
            i += 1
        return graph[:i, :]

    def compute_reachable_vertices(self, state: GameState, door_edges):
        graph = self.get_graph(state, door_edges)
        current_resources = np.array(
            [state.current_energy, state.current_missiles, state.current_super_missiles, state.current_power_bombs],
            dtype=np.int16)
        max_resources = np.array(
            [state.max_energy / state.difficulty.multiplier,
             state.max_missiles / state.difficulty.multiplier,
             state.max_super_missiles / state.difficulty.multiplier,
             state.max_power_bombs / state.difficulty.multiplier],
            dtype=np.int16)
        output = np.zeros([len(self.vertex_list), 4], dtype=np.int16)
        reachability.compute_reachability(graph, state.vertex_index, len(self.vertex_list),
                                                current_resources, max_resources, output)
        return output


sm_json_data_path = "sm-json-data/"
sm_json_data = SMJsonData(sm_json_data_path)
# for region in sm_json_data.region_json_dict.values():
#     for room in region['rooms']:
#         if 'obstacles' not in room:
#             continue
#         for obstacle in room['obstacles']:
#             if 'requires' in obstacle:
#                 print(room['name'])

# from_vertex = sm_json_data.vertex_index_dict[(38, 5, 0)]
# to_vertex = sm_json_data.vertex_index_dict[(38, 6, 1)]
# for link in sm_json_data.link_list:
#     if link.from_index == from_vertex and link.to_index == to_vertex:
#         print(link)
#         break

# difficulty_config = DifficultyConfig(
#     tech=set(),
#     shine_charge_tiles=33,
#     energy_multiplier=1.0)
# # items = {"PowerBomb", "Morph"}
# items = set()
# game_state = GameState(
#     difficulty=difficulty_config,
#     items=items,
#     flags=set(),
#     weapons=sm_json_data.get_weapons(set(items)),
#     num_energy_tanks=0,  # energy_tanks,
#     num_reserves=0,  # reserve_tanks,
#     max_energy=99,  # + 100 * (energy_tanks + reserve_tanks),
#     max_missiles=0,  # missiles,
#     max_super_missiles=0,  # super_missiles,
#     max_power_bombs=0,  # power_bombs,
#     current_energy=50,
#     current_missiles=0,  # missiles,
#     current_super_missiles=0,  # super_missiles,
#     current_power_bombs=0,  # power_bombs,
#     vertex_index=sm_json_data.vertex_index_dict[(8, 5, 0)])  # Ship (Landing Site)
#
# out = sm_json_data.compute_reachable_vertices(game_state)
# nz_i, nz_j = (out != -1).nonzero()
#
# print(nz_i.shape)
# for k in range(nz_i.shape[0]):
#     print(sm_json_data.vertex_list[nz_i[k]])
#     print(out[nz_i[k], :])

# graph = sm_json_data.get_graph(game_state)
# link.cond.get_consumption(game_state)
# link.cond.conditions[1].get_consumption(game_state)
# link.cond.conditions[1].conditions[0].get_consumption(game_state)
# link.cond.conditions[1].conditions[1].get_consumption(game_state)

# sm_json_data.
# sm_json_data.link_list[4]
# weapons = sm_json_data.get_weapons(sm_json_data.item_set)
# weapons = sm_json_data.get_weapons({"PowerBeam", "Wave", "Charge"})
# print(weapons)
# weapons
# sm_json_data.enemy_vulnerability_dict.keys()
# sm_json_data.enemy_vulnerability_dict['Kihunter (red)']

# sm_json_data.door_ptr_pair_dict
# len(sm_json_data.link_list)
