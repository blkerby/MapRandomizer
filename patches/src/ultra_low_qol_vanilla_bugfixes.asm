arch snes.cpu
lorom

;;; skips suits acquisition animation
org $848717
	rep 4 : nop

;;; fix morph ball in hidden chozo PLM
org $84e8ce
	db $04
org $84ee02
	db $04

; Use door direction ($0791) to check in Big Boy room if we are coming in from the left vs. right.
; The vanilla game instead uses layer 1 X position ($0911) in a way that doesn't work if
; door scrolling finishes before enemy initialization, a race condition which doesn't
; happen to occur in the vanilla game but can in the randomizer, for example due to a combination of 
; fast doors and longer room load time (from reloading CRE) in case we enter from Kraid's Room.
org $A9EF6C
fix_big_boy:
	LDA $0791              ; door direction
	BNE .spawn_big_boy
	LDA #$2D00			   ;\ Set enemy as intangible and invisible
	STA $0F86,x            ;/
	LDA #$EFDF             ; Enemy function = $EFDF (disappeared)
	BRA .done
.spawn_big_boy
	LDA #$EFE6             ; Enemy function = $EFE6
	NOP
org $A9EF80 
.done

; skip loading special x-ray blocks (only used in BT room during escape, and we repurpose the space for other things)
org $84836A
	BRA $2C   ; was: BEQ $2C