; Fix Choot null jump (seen in Thread the Needle Room)
; Jump calculation @ $a2dfce executed during door 
; transition can be interrupted by IRQ.

arch snes.cpu
lorom

!bank_a2_free_space_start = $a2f4a0
!bank_a2_free_space_end = $a2f4b0

org $a2dfcf
    jsr hook_multiply_entry
    
org $a2dfe2
    jsr hook_multiply_exit
    
org !bank_a2_free_space_start
hook_multiply_entry:
    sei
    sep #$20            ; replaced code
    lda $0fb4,x         ;
    rts

hook_multiply_exit:
    lda $4216           ; replaced code
    cli
    rts

warnpc !bank_a2_free_space_end
