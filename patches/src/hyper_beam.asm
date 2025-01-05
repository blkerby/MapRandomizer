;;; minimal part of escape.asm for changing Hyper Beam behavior,
;;; e.g. to use on Practice Hack for testing escape timings.
lorom
arch snes.cpu

!bank_84_free_space_start = $84F900
!bank_84_free_space_end = $84FA00

;;; carry set if escape flag on, carry clear if off
macro checkEscape()
    lda #$000e
    jsl $808233
endmacro

; Hi-jack red door Super check
org $84bd58
    jsr super_door_check
    bcs $22

; Hi-jack green door Super check
org $84bd90
    jsr super_door_check
    bcc $12

; Hi-jack yellow door PB check
org $84bd2e
    jsr pb_door_check
    bcc $12

; Hi-jack bomb block PB reaction
org $84cee8
    jsr pb_check

; Hi-jack PB block PB reaction
org $84cf3c
    jsr pb_check

; Hi-jack green gate left reaction
org $84c556
    jsr super_check

; Hi-jack green gate right reaction
org $84c575
    jsr super_check

; Hi-jack super block reaction
org $84cf75
    jsr super_check

;;; CODE in bank 84 (PLM)
org $84f900

; This function is referenced in beam_doors.asm, so it needs to be here at $84F900.
escape_hyper_door_check:
    %checkEscape() : bcc .nohit
    lda $1d77,x
    bit #$0008                  ; check for plasma (hyper = wave+plasma)
    beq .nohit
    sec                         ; set carry flag
    bra .end
.nohit:
    clc                         ; reset carry flag
.end:
    rts

;;; returns zero flag set if in the escape and projectile is hyper beam
escape_hyper_check:
    %checkEscape() : bcc .nohit
    lda $0c18,x
    bit #$0008                  ; check for plasma (hyper = wave+plasma)
    beq .nohit
    lda #$0000                  ; set zero flag
    bra .end
.nohit:
    lda #$0001                  ; reset zero flag
.end:
    rts

super_check:
    cmp #$0200                  ; vanilla check for supers
    beq .end
    jsr escape_hyper_check
.end:
    rts

super_door_check:
    pha
    cmp #$0200                  ; vanilla check for supers
    beq .end
    jsr escape_hyper_door_check
.end:
    pla
    rts

pb_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_check
.end:
    rts

pb_door_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_door_check
.end:
    rts
