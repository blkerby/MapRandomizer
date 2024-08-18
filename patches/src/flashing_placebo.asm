; Crocomire
org $A48CDD
LDX.w #$000E
-
	LDA.w CrocomireFlashPalette, X
	STA.l $7EC0E0, X
	DEX
	DEX
	BPL -

org $A4F6C0
CrocomireFlashPalette:
dw $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

; Phantoon
org $A7DD7F
LDA.w PhantoonFlashPalette, X

org $A7FFE0
PhantoonFlashPalette:
dw $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF
dw $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF
