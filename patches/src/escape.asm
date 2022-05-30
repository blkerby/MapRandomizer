;;; Based on https://raw.githubusercontent.com/theonlydude/RandomMetroidSolver/master/patches/common/src/rando_escape.asm
;;;
;;; compile with asar (https://www.smwcentral.net/?a=details&id=14560&p=section),

lorom
arch snes.cpu

;;; carry set if escape flag on, carry clear if off
macro checkEscape()
    lda #$000e
    jsl $808233
endmacro

; Set escape timer to 6 minutes (instead of 3 minutes)
org $809E20
    LDA #$0600

; Hi-jack room setup asm
org $8fe896
    jsr room_setup

; Hi-jack room main asm
org $8fe8bd
    jsr room_main

; Hi-jack load room event state header
org $82df4a
    jml music_and_enemies

; Hi-jack activate save station
org $848cf3
    jmp save_station

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
org $84f860

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

pb_check:
    cmp #$0300                  ; vanilla check for PBs
    beq .end
    jsr escape_hyper_check
.end:
    rts

;;; Disables save stations during the escape
save_station:
    %checkEscape() : bcc .end
    jmp $8d32     ; skip save station activation
.end:
    lda #$0017  ; run hi-jacked instruction
    jmp $8cf6  ; return to next instruction

org $8ff500
;;; CODE (in bank 8F free space)

room_setup:
    %checkEscape() : bcc .end
    phb                         ; do vanilla setup to call room asm
    phk
    plb
    jsr $919c                   ; sets up room shaking
    plb
    jsl fix_timer_gfx
.end:
    ;; run hi-jacked instruction, and go back to vanilla setup asm call
    lda $0018,x
    rts

room_main:
    %checkEscape() : bcc .end
    phb                         ; do vanilla setup to call room main asm
    phk
    plb
    jsr $c124                   ; explosions etc
    plb
.end:
    ;; run hi-jacked instruction, and goes back to vanilla room main asm call
    ldx $07df
    rts

fix_timer_gfx:
    PHX
    LDX $0330						;get index for the table
    LDA #$0400 : STA $D0,x  				;Size
    INX : INX						;inc X for next entry (twice because 2 bytes)
    LDA #$C000 : STA $D0,x					;source address
    INX : INX						;inc again
    SEP #$20 : LDA #$B0 : STA $D0,x : REP #$20  		;Source bank $B0
    INX							;inc once, because the bank is stored in one byte only
    ;; VRAM destination (in word addresses, basically take the byte
    ;; address from the RAM map and and devide them by 2)
    LDA #$7E00	: STA $D0,x
    INX : INX : STX $0330 					;storing index
    PLX
    RTL


music_and_enemies:
    LDA $0006,x
    STA $07CD
    %checkEscape() : bcc .end
    stz $07CB   ;} Music data index = 0
    stz $07C9   ;} Music track index = 0
.end
    jml $82df50

warnpc $8ff600