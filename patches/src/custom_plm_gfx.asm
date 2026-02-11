; custom gfx files for additional plms. these locations are unused vanilla space so it should be safe to just include this as a base patch, it will improve compatibility with multiworlds too where a non collectible walljump / spark booster / blue booster seed has the item placed in it for somebody else. Previously applied in rust but moved to simple binary files to make changing easier.

lorom

org $899100 ; wall jump boots graphics
  incbin "custom_plm_gfx/wall_jump.bin"
assert pc() <= $899200

org $899200 ; spark booster graphics
  incbin "custom_plm_gfx/spark_booster.bin"
assert pc() <= $899300

org $899300 ; blue booster graphics
  incbin "custom_plm_gfx/blue_booster.bin"
assert pc() <= $899400