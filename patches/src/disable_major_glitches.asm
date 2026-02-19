arch snes.cpu
lorom

; This patch is a consolidation of the major glitches that were previously included in the vanilla_bugfixes.asm file.
; They are relocated here to make it easier to disable them while still keeping the other quality of life features.


;;; disable GT code
;org $aac91c
;   bra $3f

; the original patch only changed the conditional branch to a forced branch.
; we can skip loading the joypad input and meaningless comparission entirely.

org $aac917
    rtl

;;; disable spacetime beam select in pause menu
;org $82b174
;	ldx #$0001

; New and improved anti-spacetime (anti-glitch beam in general)
; Hijack the suit/misc and boots routines to converge on the "beams" Main - weapons, after move response
org $82AFC4
merge_main_item_routines:
    ; lda $09A6
    ; sta $24
org $82AFC9
    bra do_button_response
category_tilemap_sizes:
    dw $0002    ; 0 = tanks - $02 just copies one word over $00 - over itself no less.
    dw $000A    ; 1 = beams
    dw $0012    ; 2 = suit/misc
    dw $0012    ; 3 = boots
assert pc() == $82AFD3
do_button_response:

org $82B0C8 ; Suit/misc - after move response call
    jmp merge_main_item_routines
; By dummying out the rest of this, we can repurpose to help with setting up the correct tilemap size value
equip_category_tilemap_size:
    ; Input: X = equipment screen category index * 2 (see button response hook for explanation)
    ; Output: $18 = tilemap size, A = $0755
    lda category_tilemap_sizes,x
    jmp continue_equip_category_tilemap_size
warnpc $82B0D2

org $82B156 ; Boots - after move response call
    jmp merge_main_item_routines
continue_equip_category_tilemap_size:
    sta $18
    lda $0755
    rts
warnpc $82B160

org $82B585
    jsr equip_category_tilemap_size