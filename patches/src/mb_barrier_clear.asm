arch snes.cpu
lorom

org $83AAD2
    dw $EB00  ; Set door ASM for Rinka Room toward Mother Brain

org $8FEB00
    ; clear barriers in mother brain room based on main bosses killed:
    ; clear kraid barrier
    lda $7ed829
    bit #$0001
    beq phantoon  ; skip clearing if kraid isn't dead

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm

    ; clear phantoon barrier
phantoon:
    lda $7ed82b
    and #$0001
    beq draygon  ; skip clearing if phantoon isn't dead

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm

    ; clear draygon barrier
draygon:
    lda $7ed82c
    bit #$0001
    beq ridley  ; skip clearing if draygon isn't dead

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm

    ; clear ridley barrier
ridley:
    lda $7ed82a
    bit #$0001
    beq motherbrain  ; skip clearing if ridley isn't dead

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm

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


org $83AAEA
    dw $EE00  ; Set door ASM for Tourian Escape Room 1 toward Mother Brain

org $83AAE3
    db $00    ; Set door direction = $00  (to make door not close behind Samus)

; Custom door ASM for Tourian Escape Room 1 toward Mother Brain
org $8FEE00
    jsl $8483D7            ;\
    db  $00, $06           ;|
    dw  $B677              ;} Spawn Mother Brain's room escape door

    ; Remove invisible spikes where Mother Brain used to be:
    jsl remove_spikes

    rts

; Remove invisible spikes where Mother Brain used to be (common routine used by both the left and right door ASMs)
org $84F200
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

warnpc $84F300

;org $82E801
;    JSR after_level_data_load
;
;org $82EAA9
;    JSR after_level_data_load
;
;org $82FA00
;after_level_data_load:
;    INC $09C8
;    LDA #$0000  ; run hi-jacked instruction
;    RTS

;org $82E725
;    JSL end_door_transition
;
;org $82FA00      ; TODO: add this to rom map
;end_door_transition:
;    INC $09C8
;    jsl $80A149
;;    jsl $80A176   ; Display the viewable part of the room
;    JSL $908E0F   ; run hi-jacked instruction
;    RTL



