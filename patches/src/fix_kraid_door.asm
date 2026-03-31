; Fix bug when entering Kraid from right door and leaving just as he
; finishes ascending. The boss AI checks that Samus is left of Kraid,
; and will adjust her X coordinate if needed. This can lead to a bugged
; Samus position or OOB error if it occurs during the door transition.
;
; Reported by P.JBoy, fix by Stag Shot

lorom

!bank_a7_free_space_start = $a7ffc0
!bank_a7_free_space_end = $a7ffd0

org $a7b0fb
    jmp  fix_kraid_door

org !bank_a7_free_space_start
fix_kraid_door:
    lda $795        ; door transition flag
    beq .resume
    rts             ; skip rest of func

.resume
    lda $af6        ; replaced code
    clc             ;
    jmp $b0ff       ; resume func

assert pc() < !bank_a7_free_space_end
