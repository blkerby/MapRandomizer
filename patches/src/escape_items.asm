; hi-jack collecting hyper beam
org $A9CD12
    jsl get_hyper_beam

; free space in bank $A9, but must match address in patch.rs (for collecting hyper beam with fast Mother Brain)
org $A9FB70
get_hyper_beam:
    jsl $91E4AD   ; run the hi-jacked instruction
    lda #$F72F
    sta $09A2   ; all items equipped (including WallJump = $0400)
    sta $09A4   ; all items collected

    ; Clear selected HUD item
    lda #$0000
    sta $09D2

    jsl $809A2E   ; Add grapple to HUD tilemap
    jsl $809A3E   ; Add x-ray to HUD tilemap

    rtl
assert pc() <= $A9FC00