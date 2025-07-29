!bank_84_free_space_start = $84F5C0
!bank_84_free_space_end = $84F630

org !bank_84_free_space_start

horizontal_wall:
    dw $B3D0, horizontal_wall_inst

vertical_wall_2:
    dw $B3D0, vertical_wall_2_inst

vertical_wall_3:
    dw $B3D0, vertical_wall_3_inst

vertical_wall_4:
    dw $B3D0, vertical_wall_4_inst

horizontal_wall_inst:
    dw $0001, horizontal_wall_draw
    dw $86BC

vertical_wall_2_inst:
    dw $0001, vertical_wall_2_draw
    dw $86BC

vertical_wall_3_inst:
    dw $0001, vertical_wall_3_draw
    dw $86BC

vertical_wall_4_inst:
    dw $0001, vertical_wall_4_draw
    dw $86BC

horizontal_wall_draw:
    dw $8004, $805F, $805F, $805F, $805F
    db $01, $00
    dw $8004, $805F, $805F, $805F, $805F
    dw $0000

; We save space by overlapping the draw instructions together
; for the 2-tile, 3-tile, and 4-tile thick vertical walls:
vertical_wall_4_draw:
    dw $0004, $805F, $805F, $805F, $805F
    db $00, $03
vertical_wall_3_draw:
    dw $0004, $805F, $805F, $805F, $805F
    db $00, $02
vertical_wall_2_draw:
    dw $0004, $805F, $805F, $805F, $805F
    db $00, $01
    dw $0004, $805F, $805F, $805F, $805F
    dw $0000

print pc
warnpc !bank_84_free_space_end