!bank_84_free_space_start = $84F5A0
!bank_84_free_space_end = $84F5C0

; hook checks for if energy is already full (to make it check both regular and reserve energy)
org $848CB4
    jsr check_full    ; replaces CMP $09C2 

org $84AE38
    jsr check_full    ; replaces CMP $09C2 

org $84B285
    ; replaces LDA $09C2 : CMP $09C4
    lda $09C4
    jsr check_full

org $84B2B8
    ; replaces LDA $09C2 : CMP $09C4
    lda $09C4
    jsr check_full

; hook code to fill energy when using an energy station:
; replaces STA $09C2
org $848CC3
    jsr fill_energy

org !bank_84_free_space_start
check_full:
    cmp $09C2  ; current regular energy == max regular energy?
    bne .end
    lda $09D4  ; max reserve energy
    cmp $09D6  ; current reserve energy
.end
    ; Return to vanilla code. If zero flag is set, it indicates to skip the refill.
    rts

fill_energy:
    sta $09C2  ; current regular energy <- max regular energy
    lda $09D4  ; max reserve energy
    sta $09D6  ; current reserve energy <- max
    rts

warnpc !bank_84_free_space_end