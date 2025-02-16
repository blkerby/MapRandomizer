arch snes.cpu
lorom

!bank_84_freespace_start = $84F200
!bank_84_freespace_end = $84F300

incsrc "constants.asm"

macro clear_plm(plm)
    jsl $8483D7
    db <plm>
    db $04
    dw clear_barrier_plm
endmacro

org $83AAD2
    dw $EB00  ; Set door ASM for Rinka Room toward Mother Brain

; Free space in bank $8F
org $8FEB00
door_asm_start:
; 3 potential scenarios for barriers:
; obj_num < 4, clear from left to right
; obj_num = 4, stock behavior
; obj_num > 4, maintain all 4 until all obj's cleared
    lda !objectives_num : and $7FFF
    beq clear_all                 ; None ?
    cmp #$0005
    bcc normal_objs               ; <= 4 ?

    ldx #$0000
    tay
; iterate through all obj's
.check_lp
    jsr check_objective
    beq motherbrain
    dey
    bne .check_lp

; either none or all cleared
clear_all:
    ldx #$FFFB
    bra start_obj_checks

; # obj's - 4
normal_objs:
    sec
    sbc #$0004
    tax

; starting value of X determines check/clear behavior
; X = 0 for stock behavior (4 objs)
; X < 0 will clear until 0, then check (for 0-3 objs)
; X = -4 will clear all objs
start_obj_checks:
    jsr check_objective
    beq .skip_1
    %clear_plm($36)
.skip_1
    jsr check_objective
    beq .skip_2
    %clear_plm($37)
.skip_2
    jsr check_objective
    beq .skip_3
    %clear_plm($38)
.skip_3
    jsr check_objective
    beq motherbrain
    %clear_plm($39)

motherbrain:
    lda $7ed82d
    bit #$0001
    beq done  ; skip clearing if mother brain isn't dead

    ; Spawn Mother Brain's room escape door:
    jsl $8483D7
    dw  $0600,  $B677

    ; Remove invisible spikes where Mother Brain used to be:
    jsl remove_spikes
done:
    rts
    
check_objective:
; X >= 0 obj to check; X < 0 = return obj cleared
; also increments X
    phx
    txa
    bmi .bypass_check
    asl
    tax
    lda.w #$007E
    sta.b $02
    lda.l ObjectiveAddrs, X
    sta.b $00
    lda.l ObjectiveBitmasks, X
    plx
    inx
    sta.b $04
    lda.b [$00]
    bit.b $04
    rts
.bypass_check
    plx
    inx
    lda #$0001
    rts

warnpc !objectives_addrs

org !objectives_addrs
ObjectiveAddrs:
    dw $D829, $D82A, $D82B, $D82C, $D82D
    dw $D82E, $D82F, $D830, $D831, $D832
    dw $D833, $D834, $D835, $D836, $D837
    dw $D838, $D839, $D83A, $D83B, $D83C
    dw $D83D, $D83E, $D83F, $D840, $D841
ObjectiveBitmasks:
    dw $0001, $0001, $0001, $0001, $0002
    dw $0001, $0001, $0001, $0001, $0001
    dw $0001, $0001, $0001, $0001, $0001
    dw $0001, $0001, $0001, $0001, $0001
    dw $0001, $0001, $0001, $0001, $0001

; OBJECTIVE: None (must match address in patch.rs)
warnpc $8FECA0
org $8FECA0
    jmp clear_all ; this section can be removed once patch.rs is updated
    
warnpc $8FED00

; Remove invisible spikes where Mother Brain used to be (common routine used by both the left and right door ASMs)
org !bank_84_freespace_start

remove_spikes:
    ; Remove invisible spikes
    lda #$8000   ; solid tile
    ldx #$0192   ; offset to spike above Mother Brain right
    jsr $82B4
    lda #$8000   ; solid tile
    ldx #$0210   ; offset to spike above Mother Brain center-right
    jsr $82B4
    lda #$8000   ; solid tile
    ldx #$0494   ; offset to spike below Mother Brain right
    jsr $82B4
    rtl

clear_barrier_plm:
    dw $B3D0, clear_barrier_inst

clear_barrier_inst:
    dw $0001, clear_barrier_draw
    dw $86BC

clear_barrier_draw:
    dw $8006, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0000

bowling_chozo_set_flag:
    jsl $90F084  ; run hi-jacked instruction
    lda $7ed823  ; set flag to represent that Bowling statue has been used (flag bit unused by vanilla game)
    ora #$0001
    sta $7ed823
    rtl

pirates_set_flag:
    jsl $8081FA  ; run hi-jacked instruction (set zebes awake flag)

    lda $079B   ; room pointer    
    cmp #$975C  ; pit room?
    bne .not_pit_room
    lda $7ED823
    ora #$0002
    sta $7ED823
    rtl
.not_pit_room:
    cmp #$A521  ; baby kraid room?
    bne .not_baby_kraid_room
    lda $7ED823
    ora #$0004
    sta $7ED823
    rtl
.not_baby_kraid_room:
    cmp #$D2AA  ; plasma room?
    bne .not_plasma_room
    lda $7ED823
    ora #$0008
    sta $7ED823
    rtl
.not_plasma_room:
    cmp #$B62B  ; metal pirates room?
    bne .not_metal_pirates_room
    lda $7ED823
    ora #$0010
    sta $7ED823
.not_metal_pirates_room:
    rtl

warnpc !bank_84_freespace_end

; bowling chozo hook
org $84D66B
    jsl bowling_chozo_set_flag

; enemies (pirates) dead hook
org $84BE0E
    jsl pirates_set_flag
