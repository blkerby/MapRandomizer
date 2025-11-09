arch snes.cpu
lorom

!bank_84_free_space_start = $84F200
!bank_84_free_space_end = $84F300

!bank_b8_free_space_start = $B88100   ; must match address in patch.asm
!bank_b8_free_space_end = $B88300

incsrc "constants.asm"

macro clear_plm(plm)
    jsl $8483D7
    db <plm>
    db $04
    dw clear_barrier_plm
endmacro

; Hook the code that checks whether to trigger post-MB1 cutscene:
; (based on glass being broken, MB1 health at zero, etc.)
org $A987E4
    jsl mb1_trigger_hook

; Free space in bank $B8
org !bank_b8_free_space_start
; This must be placed at the start of the free space, to match the location expected in patch.rs
door_asm_start:
; 2 potential scenarios for barriers:
; obj_num <= 4, clear individually from left to right
; obj_num > 4, maintain all 4 until all obj's cleared
    lda !objectives_num : and #$7FFF
    cmp #$0005
    bcc normal_objs               ; <= 4 ?

    ldx #$0000
    tay
; iterate through all obj's
.check_lp
    jsr check_objective
    beq done
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
    %clear_plm($39)
.skip_1
    jsr check_objective
    beq .skip_2
    %clear_plm($38)
.skip_2
    jsr check_objective
    beq .skip_3
    %clear_plm($37)
.skip_3
    jsr check_objective
    beq done
    %clear_plm($36)

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
    lda.l !objectives_addrs, X
    sta.b $00
    lda.l !objectives_bitmasks, X
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

; returns: carry set if conditions to trigger MB1 cutscene are satisfied
; (glass broken and all objectives complete)
mb1_trigger_hook:
    jsl $808233  ; run hi-jacked code: check if glass is broken
    bcc .done

    lda !objectives_num : and #$7FFF
    beq .all_obj_cleared

    ldx #$0000
    tay
.check_lp:
    jsr check_objective
    beq .obj_not_cleared
    dey
    bne .check_lp
.all_obj_cleared:
    sec
    rtl
.obj_not_cleared:
    clc
.done:
    rtl
warnpc !bank_b8_free_space_end

org !bank_84_free_space_start

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

warnpc !bank_84_free_space_end

; bowling chozo hook
org $84D66B
    jsl bowling_chozo_set_flag

; enemies (pirates) dead hook
org $84BE0E
    jsl pirates_set_flag
