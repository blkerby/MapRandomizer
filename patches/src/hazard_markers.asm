lorom

!bank_84_free_space_start = $84F580
!bank_84_free_space_end = $84F600
!bank_8f_free_space_start = $8FFE80
!bank_8f_free_space_end = $8FFF00

; landing site: testing using setup ASM
org $8F922B : dw spawn_right_hazard
org $8F9245 : dw spawn_right_hazard
org $8F925F : dw spawn_right_hazard
org $8F9279 : dw spawn_right_hazard

org $82E7A8
    jsl load_hazard_tiles

org !bank_8f_free_space_start

spawn_right_hazard:
    JSL $88A7D8  ; vanilla setup ASM (scrolling sky)

    jsl $8483D7
    db $8F
    db $46
    dw right_hazard_plm
    rts

warnpc !bank_8f_free_space_end

org !bank_84_free_space_start

load_hazard_tiles:
    jsl $80B271  ; run hi-jacked instruction (decompress CRE tiles from $B98000 to VRAM $2800)

    

    rtl

right_hazard_plm:
    dw $B3D0, right_hazard_inst

right_hazard_inst:
    dw $0001, right_hazard_draw
    dw $86BC

right_hazard_draw:
    dw $8004, $00AA, $00CA, $08CA, $08AA, $0000

warnpc !bank_84_free_space_end