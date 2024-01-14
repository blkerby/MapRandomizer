lorom

; remove check for >=30 energy to continue shinespark
org $90D2BD 
    cmp #$0000   ; replaces CMP #$001E

; remove energy decrement: vertical shinespark
org $90D0CE
    nop : nop : nop   ; replaces DEC $09C2

; remove energy decrement: diagonal shinespark
org $90D0FD
    nop : nop : nop   ; replaces DEC $09C2

; remove energy decrement: horizontal shinespark
org $90D129
    nop : nop : nop   ; replaces DEC $09C2
