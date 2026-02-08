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