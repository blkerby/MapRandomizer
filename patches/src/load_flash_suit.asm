!bank_80_free_space_start = $80E660
!bank_80_free_space_end = $80E670

org $80C461
    JSR check_flash_suit

; the saving of the flash/blue suit state is handled in "saveload.asm":
; after loading from a save, $7EFE90 bit 0 (bitmask #$0001) marks flash suit,
; bit 1 (bitmask #$0002) marks blue suit
org !bank_80_free_space_start
check_flash_suit:
    LDA $7EFE90
    AND #$0001
    STA $0A68   ; special samus palette timer <- 1 if flash suit stored, otherwise 0
    LDA $0002,x ; run hi-jacked instruction
    RTS
warnpc !bank_80_free_space_end
