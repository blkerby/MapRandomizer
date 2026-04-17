Files here are lists of address ranges used by patches or reserved for use by the randomiser. Free space usage is governed by `Bank N.txt` files, vanilla overwrites are governed by `vanilla_hooks.txt`.

Patch usage can be generated from the files in `../ips/` by going to `../../python/scripts/` and running `dump_rom_map.py`

The format of list entries is:
* `89AB - CDEF: ; patch_filename.asm` - for space used by an IPS patch. These may be added, removed or modified by `dump_rom_map.py`
* `89AB - CDEF: Description of address range` - for space reserved for use. These lines are preserved by `dump_rom_map.py`

Where both `89AB` and `CDEF` are inclusive (so `8000 - FFFF` is the whole bank).
