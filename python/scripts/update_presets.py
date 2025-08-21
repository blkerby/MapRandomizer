import glob
import json

skill_presets_path = "rust/data/presets/skill-assumptions"
item_presets_path = "rust/data/presets/item-progression"
qol_presets_path = "rust/data/presets/quality-of-life"
full_settings_path = "rust/data/presets/full-settings"

skill_presets = {}
for path in glob.glob(skill_presets_path + "/*.json"):
    preset = json.load(open(path, "r"))
    skill_presets[preset["preset"]] = preset

item_presets = {}
for path in glob.glob(item_presets_path + "/*.json"):
    preset = json.load(open(path, "r"))
    item_presets[preset["preset"]] = preset

qol_presets = {}
for path in glob.glob(qol_presets_path + "/*.json"):
    preset = json.load(open(path, "r"))
    qol_presets[preset["preset"]] = preset

version = int(open("rust/VERSION", "r").read())
print("Version:", version)

# Update full-settings presets:
for path in glob.glob(full_settings_path + "/*.json"):
    settings = json.load(open(path, "r"))

    skill_preset_name = settings["skill_assumption_settings"]["preset"]
    if skill_preset_name is not None:
        settings["skill_assumption_settings"] = skill_presets[skill_preset_name]

    item_preset_name = settings["item_progression_settings"]["preset"]
    if item_preset_name is not None:
        settings["item_progression_settings"] = item_presets[item_preset_name]

    qol_preset_name = settings["quality_of_life_settings"]["preset"]
    if qol_preset_name is not None:
        settings["quality_of_life_settings"] = qol_presets[qol_preset_name]

    settings["version"] = version
    json.dump(settings, open(path, "w"), indent=4)
