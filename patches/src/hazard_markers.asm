lorom

incsrc "constants.asm"

!bank_84_free_space_start = $84F800   ; must match address in patch.rs
!bank_84_free_space_end = $84F900


!hazard_tilemap_size = #$0020

org $82E845
    jsl load_hazard_tilemap_initial_hook
    nop #3

; hook door transition
org $82EB20
    jsl reload_hazard_tiles
    nop : nop

org !bank_84_free_space_start

; These PLMs definitions must go here first, as they are referenced in patch.rs
right_hazard_transition_plm:
    dw $B3D0, right_hazard_transition_inst

down_hazard_plm:
    dw $B3D0, down_hazard_inst

down_hazard_transition_plm:
    dw $B3D0, down_hazard_transition_inst

left_hazard_transition_plm:
    dw $B3D0, left_hazard_transition_inst

elevator_hazard_plm:
    dw $B3D0, elevator_hazard_inst

elevator_hazard_with_scroll_plm:
    dw $B3D0, elevator_hazard_with_scroll_inst

load_hazard_tilemap_initial_hook:
    JSL $80B0FF  ; run hi-jacked instruction (Decompress CRE tile table to $7E:A000)
    dl $7EA000
; falls through to below:
load_hazard_tilemap:
    lda $079B
    cmp #$A59F  ; is this Kraid Room?
    bne .skip

    ; The Kraid Room SCE tileset overwrites the CRE, including hazard tiles. So we have to manually
    ; point the tilemap to a different copy of the hazard tiles.
    lda #$227C
    sta $7EA700
    lda #$227D
    sta $7EA704
    lda #$227E
    sta $7EA708
    lda #$227F
    sta $7EA70C

.skip:
    rtl

reload_hazard_tiles:
    ; Copy hazard tilemap (definition of 16 x 16 tiles from 8 x 8 tiles)
    jsl load_hazard_tilemap

    ; run-hijacked instructions
    ldx $07BB
    ldy $000E,x

    rtl

right_hazard_transition_inst:
    dw $0001, right_hazard_transition_draw
    dw $86BC

right_hazard_transition_draw:
    dw $8004, $90E0, $90E1, $98E1, $98E0, $0000

left_hazard_transition_inst:
    dw $0001, left_hazard_transition_draw
    dw $86BC

left_hazard_transition_draw:
    dw $8004, $94E0, $94E1, $9CE1, $9CE0, $0000

down_hazard_inst:
    dw $0001, down_hazard_draw
    dw $86BC

down_hazard_draw:
    dw $0004, $00E2, $00E3, $04E3, $04E2, $0000

down_hazard_transition_inst:
    dw $0001, down_hazard_transition_draw
    dw $86BC

down_hazard_transition_draw:
    dw $0004, $90E2, $90E3, $94E3, $94E2, $0000

elevator_hazard_inst:
    dw $0001, elevator_hazard_draw
    dw $86BC

elevator_hazard_draw:
    dw $0002, $90E4, $94E4
    db $00, $01
    dw $0002, $00E5, $04E5
    dw $0000

elevator_hazard_with_scroll_inst:
    dw $0001, elevator_hazard_with_scroll_draw
    dw $86BC

elevator_hazard_with_scroll_draw:
    dw $0002, $90E4, $94E4
    db $00, $01
    dw $0002, $00E5, $34E5  ; use special air block type, to preserve scroll PLM
    dw $0000

assert pc() <= !bank_84_free_space_end