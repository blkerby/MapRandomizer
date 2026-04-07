; From https://github.com/aremath/sm_rando/blob/master/patches/asm/mother_brain_no_drain.asm
; This patches Mother Brain's Rainbow beam attack to not drain Samus' ammo.
lorom

org $A9C544
dw #$0000
