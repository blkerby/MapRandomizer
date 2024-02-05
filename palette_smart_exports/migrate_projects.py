import os
import shutil

project_mapping = [
    ("crateria", "CrateriaPalette"),
    # ("brinstar", "BrinstarPalette"),
    # ("norfair", "NorfairPalette"),
    # ("wrecked_ship", "WreckedShipPalette"),
    # ("maridia", "MaridiaPalette"),
    # ("tourian", "TourianPalette")
]
tileset_mapping = [
    ("00", "00"),
    ("01", "01"),
    ("02", "02"),
    ("03", "03"),
    ("04", "04"),
    ("05", "05"),
    ("05", "1E"),  # Phantoon
    ("06", "06"),
    ("06", "1D"),  # SpoSpo
    ("07", "07"),
    ("08", "08"),
    ("09", "09"),
    ("0A", "0A"),
    ("0B", "0B"),
    ("0C", "0C"),
    ("0D", "0D"),
    ("0E", "0E"),
    ("0F", "0F"),
    ("10", "10"),
    ("11", "11"),
    ("12", "12"),
    ("13", "13"),
    ("14", "14"),
    ("15", "15"),
    ("16", "16"),
    ("17", "17"),
    ("18", "18"),
    ("19", "19"),
    ("1A", "1A"),
    ("1B", "1B"),
    ("1C", "1C"),
]
old_path = "./"
new_path = "../../Mosaic/Projects/"
base_path = new_path + "Base/"

for old_project_name, new_project_name in project_mapping:
    old_project_path = old_path + old_project_name + "/"
    new_project_path = new_path + new_project_name + "/"
    print(old_project_path, "->", new_project_path)

    os.unlink(new_project_path + "Export/Tileset")
    os.mkdir(new_project_path + "Export/Tileset")
    os.symlink("..\\..\\..\\Base\\Export\\Tileset\\CRE", new_project_path + "Export/Tileset/CRE", True)
    os.mkdir(new_project_path + "Export/Tileset/SCE")
    for old_tileset_str, new_tileset_str in tileset_mapping:
        os.mkdir(new_project_path + "Export/Tileset/SCE/" + new_tileset_str);
        base_tileset_path = "..\\..\\..\\..\\..\\Base\\Export\\Tileset\\SCE\\" + new_tileset_str + "\\"
        new_tileset_path = new_project_path + "Export/Tileset/SCE/" + new_tileset_str + "/"
        os.symlink(base_tileset_path + "8x8tiles.gfx", new_tileset_path + "8x8tiles.gfx")
        os.symlink(base_tileset_path + "16x16tiles.ttb", new_tileset_path + "16x16tiles.ttb")
        shutil.copy(old_project_path + f"Export/Tileset/SCE/{old_tileset_str}/palette.snes", new_tileset_path + "palette.snes")
