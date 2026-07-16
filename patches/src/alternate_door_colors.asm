arch snes.cpu
lorom

!bank_82_free_space_start = $82FA00 
!bank_82_free_space_end = $82FA80

!bank_df_free_space_start = $dfe200 ; must match addresses in customize.rs [these colors may be modifided there]
!bank_df_free_space_end = $dfe212

org $82E7D0
    jmp hook_load_tileset

org $82E65D
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
    jsr update_palette
    lda #$E664
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
    lda.l pb_1
    sta $C002
    sta $C202
    lda.l pb_2
    sta $C004
    sta $C204
    lda.l pb_3
    sta $C006
    sta $C206
    ; Super door:
    lda.l super_1
    sta $C022
    sta $C222
    lda.l super_2
    sta $C024
    sta $C224
    lda.l super_3
    sta $C026
    sta $C226
    ; Missile door:
    lda.l missile_1
    sta $C042
    sta $C242
    lda.l missile_2
    sta $C044
    sta $C244
    lda.l missile_3
    sta $C046
    sta $C246
    
    plb
    rts

assert pc() <= !bank_82_free_space_end

org !bank_df_free_space_start

pb_1:
  dw $019E
pb_2:
  dw $0114
pb_3:
  dw $008A
super_1:
  dw $43F0
super_2:
  dw $2A8A
super_3:
  dw $1184
missile_1:
  dw $7C1F
missile_2: 
  dw $5816
missile_3:
  dw $340D

assert pc() <= !bank_df_free_space_end