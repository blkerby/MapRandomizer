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
org $82b174
	ldx #$0001