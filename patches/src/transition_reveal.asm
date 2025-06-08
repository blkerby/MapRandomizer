; Helper function to reveal map tiles through door transitions.
; This is called in custom door ASM to show area transition markers and elevators.

!bank_8f_freespace_start = $8FEDC0  ; address must match reference in patch.rs
!bank_8f_freespace_end = $8FEE00

org !bank_8f_freespace_start
; input: 16-bit X = offset into map revealed data (relative to $702000)
;        16-bit A = bitmask for specific revealed bit
reveal_tile:
    ; set map revealed bit
    pha
    ora $702000,X
    sta $702000,X
    pla

    ; set map partial revealed bit
    ora $702700,X
    sta $702700,X
    rts

warnpc !bank_8f_freespace_end
