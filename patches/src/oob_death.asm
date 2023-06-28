; Triggers death if Samus goes out of bounds

arch snes.cpu
lorom

!any_bank_free_space_start = $80D140
!any_bank_free_space_end = $80D200

; hook the end of main gameplay
;org $828BB3
org $828BAF
    jsl check_oob

org !any_bank_free_space_start
check_oob:
    ;jsl $A09169  ; run hi-jacked instruction
    jsl $A08687
    lda #$0012
    cmp $0998
    bcc .skip    ; skip check if we are already in a death animation state

    lda $0AF6  ; samus X position
    sec
    sbc $0AFE  ; A <- samus X position - samus X radius
    bmi .oob   ; OOB if A is negative (Samus is beyond left edge of room)

    lda $0AF6  ; samus X position
    clc
    adc $0AFE  ; A <- samus X position + samus X radius
    dec
    xba
    and #$00FF
    cmp $07A9
    bcs .oob   ; OOB if Samus is beyond right edge of room

    lda $0AFA  ; samus Y position
    sec
    sbc $0B00  ; A <- samus Y position - samus Y radius
    bmi .oob   ; OOB if A is negative (Samus is beyond top edge of room)

    lda $0AFA  ; samus Y position
    clc
    adc $0B00  ; A <- samus Y position + samus Y radius
    dec
    xba
    and #$00FF
    cmp $07AB
    bcs .oob   ; OOB if Samus is beyond bottom edge of room

    rtl

.oob:
    ; check if we're on an elevator ride. Samus normally goes OOB on elevators and we don't want to trigger death in that case.
    lda $0E18
    bne .skip

    ; check if we're grabbed by Draygon. Samus can easily go OOB while grabbed by Draygon (while morphed) and we don't want to trigger death in that case.
    lda $0A1F
    and #$00FF
    cmp #$001A
    beq .skip

    lda #$0013
    sta $0998    ; set game state = $13 (death sequence start)
.skip:
    rtl
warnpc !any_bank_free_space_end