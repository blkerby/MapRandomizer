arch snes.cpu
lorom

; Hijack code to test for full ammo before activating Missile Refill. Instead of only checking for full missile ammo,
; we check for full ammo of all three types.
org $848CD2
    jsr check_full      ; overwrite vanilla instruction LDA $09C8
    nop : nop : nop     ; overwrite vanilla instruction CMP $09C6

org $84AEBF
    jsr check_full      ; overwrite vanilla instruction LDA $09C8
    nop : nop : nop     ; overwrite vanilla instruction CMP $09C6

org $84B2E8
    jsr check_full     ; overwrite vanilla instruction LDA $09C6
    nop : nop : nop    ; overwrite vanilla instruction CMP $09C8

org $84B31B
    jsr check_full     ; overwrite vanilla instruction LDA $09C6
    nop : nop : nop    ; overwrite vanilla instruction CMP $09C8

; Hijack code to fill ammo.
org $848CE1
    jsr fill_ammo      ; overwrite vanilla instruction LDA $09C8
    nop : nop : nop    ; overwrite vanilla instruction STA $09C6

; Free space in bank $84 for our new code:
org $84F100

; Check if all three ammo types are full (in which case the refill should be skipped):
check_full:
    lda $09C6  ; current missiles
    cmp $09C8  ; max missiles
    bne .end
    lda $09CA  ; current supers
    cmp $09CC  ; max supers
    bne .end
    lda $09CE  ; current power bombs
    cmp $09D0  ; max power bombs
.end
    ; Return vanilla code. If zero flag is set, it indicates to skip the refill.
    rts

fill_ammo:
    lda $09C8
    sta $09C6
    lda $09CC
    sta $09CA
    lda $09D0
    sta $09CE
    rts

warnpc $84F200