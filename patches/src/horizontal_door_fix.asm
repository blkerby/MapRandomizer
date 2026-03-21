; Horizontal door duplication bugfix
;
; Fixes rare instance (seen with vanilla HUD & pink Brinstar theme at Warehouse entrance)
; where a door can be drawn at upper corner after transition. This occurs due to a 
; HUD IRQ triggering in the middle of the PLM block calculation @ $84848e.
;
; Bug found/characterized by somerando, patch by Maddo/Stag Shot


arch 65816
lorom

!bank_84_free_space_start = $84f590
!bank_84_free_space_end = $84f5a0

org $848488
    jmp clr_ints
clr_ret:

org $8484e0
    jmp set_ints
set_ret:

org !bank_84_free_space_start
clr_ints:
    sei           ; disable IRQ
    sta $4202     ; replaced code
    jmp clr_ret

set_ints:
    cli           ; enable IRQ
    plx           ; replaced code
    ply
    plb
    jmp set_ret

warnpc !bank_84_free_space_end
