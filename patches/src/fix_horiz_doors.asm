; Fix rare layer 2 graphical glitch where the columns behind the door appears before being
; updated. Fix is to adjust horizontal door scrolling BG2 code to start at column door-1.
; - Stag Shot

arch 65816

!bank_80_freespace_start = $80E180
!bank_80_freespace_end = $80E1B0

org $80ad4e
     jsr check_door_r

org $80ad78
     jsr check_door_l

org $80aea2
    jsr fix_scroll_r
    
org $80aee6
    jsr fix_scroll_l

org !bank_80_freespace_start
check_door_r:
    sbc #$0104  ; adjust starting X position
    rts

check_door_l:
    adc #$0104  ; adjust starting X position
    rts

fix_scroll_r:
    lda $917    ; replaced code
    ldy $925
    cpy #$003f
    bmi .leave
    clc
    adc #$0004 ; double last scroll adjustment
.leave
    rts

fix_scroll_l:
    lda $917    ; replaced code
    ldy $925
    cpy #$003f
    bmi .leave
    sec
    sbc #$0004 ; double last scroll adjustment
.leave
    rts

warnpc !bank_80_freespace_end
