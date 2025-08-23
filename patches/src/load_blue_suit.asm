!bank_80_free_space_start = $80E670
!bank_80_free_space_end = $80E690

org $80C467
    JSR check_blue_suit

; the saving of the flash/blue suit state is handled in "saveload.asm":
; after loading from a save, $7EFE90 bit 0 (bitmask #$0001) marks flash suit,
; bit 1 (bitmask #$0002) marks blue suit
org !bank_80_free_space_start
check_blue_suit:
    LDA $7EFE90
    AND #$0002
    BEQ .skip
    LDA #$0400
    STA $0B3E   ; set dash counter to 4
.skip:
    LDA $0004,x ; run hi-jacked instruction
    RTS
warnpc !bank_80_free_space_end
