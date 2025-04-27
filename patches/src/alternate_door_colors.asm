!bank_82_free_space_start = $82FA00
!bank_82_free_space_end = $82FA80

!pb_1 = #$019E
!pb_2 = #$0114
!pb_3 = #$008A
!super_1 = #$43F0
!super_2 = #$2A8A
!super_3 = #$1184
!missile_1 = #$7C1F
!missile_2 = #$5816
!missile_3 = #$340D

org $82E7D0
    jmp hook_load_tileset

org $82E4A5
    jsr hook_door_transition

org $828D2C
    jsr hook_pause

org !bank_82_free_space_start
hook_load_tileset:
    jsr update_palette
    plb
    plp
    rtl

hook_door_transition:
    sta $099C
    jsr update_palette
    rts

hook_pause:
    jsr update_palette
    jmp $9009  ; jump to hi-jacked routine

update_palette:
    phb
    pea $7E7E
    plb
    plb
    ; Power Bomb door:
    lda !pb_1
    sta $C002
    sta $C202
    lda !pb_2
    sta $C004
    sta $C204
    lda !pb_3
    sta $C006
    sta $C206
    ; Super door:
    lda !super_1
    sta $C022
    sta $C222
    lda !super_2
    sta $C024
    sta $C224
    lda !super_3
    sta $C026
    sta $C226
    ; Missile door:
    lda !missile_1
    sta $C042
    sta $C242
    lda !missile_2
    sta $C044
    sta $C244
    lda !missile_3
    sta $C046
    sta $C246
    
    plb
    rts

warnpc !bank_82_free_space_end