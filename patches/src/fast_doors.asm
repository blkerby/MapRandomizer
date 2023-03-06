; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/fast_doors.asm

arch snes.cpu

;;; add Kejardon faster decompression patch (has lorom directive)
; incsrc "decompression.asm"

;;; Door Transition Speed by kazuto + extra code to fix samus initial position
;;; just double the speed, discard configurability due to the extra code

;Door stuff
!S = $0008	;Door scroll speed

;Don't touch these values
!X = $0100/!S	;Horizontal loop counter, X times S should equal $100
!Y = $00E0/!S+1	;Vertical loop counter, Y times S should equal $E0 ($100-$20 due to the HUD)
!C = $0010/!S+1	;Vertical counter for drawing tile rows "behind" HUD (prior to scrolling)

!direction = $0791
!layer1_x = $0911
!layer1_y = $0915
!samus_x = $0af6
!samus_sx = $0af8
!samus_y = $0afa
!samus_sy = $0afc

org $80AE9D
	dw !S
org $80AEA7
	dw !S
org $80AEB6
	dw !X
org $80AEE1
	dw !S
org $80AEEB
	dw !S
org $80AEFA
	dw !X
org $80AF45
	dw !Y
org $80AF64
	dw !S
org $80AF6E
	dw !S
org $80AF7D
	dw !Y
org $80AFE6
	dw !S
org $80AFF0
	dw !S
org $80AFF6
	dw !C
org $80B02A
	dw !Y

org $82DE50
	BPL $0F
	LDA !direction
	ROR A
	ROR A
	BCS $05
	LDA #$00C8
	BRA $03
	LDA #$0180

;Uncomment one of the three following lines
;	LSR A		;Uncomment only if S equals $0002
;	NOP		;Uncomment only if S equals $0004
	ASL A		;Uncomment only if S equals $0008

;;; Door centering speed by Kazuto:
!Speed = 2	;Pixels per-frame to slide the screen, default $01
!FreeSpace = $82F900	; Needs to be in bank $82

org $82E325	;Horizontal doors
	NOP
	PHP
	LDX #$0004
	PLP
	JSL SlideCode

org $82E339	;Vertical doors
	NOP
	PHP
	LDX #$0000
	PLP
	JSL SlideCode

org $82E2DB
	JSR set_fadeout

org !FreeSpace
SlideCode:
	SEP #$20
	BMI $09
	LDA !layer1_x,X
	CMP.b #$00+!Speed
	BPL $12
	BRA $0A
	LDA !layer1_x,X
	CMP.b #$00-!Speed
	BMI $10
	INC !layer1_x+1,X	;These two lines handle odd
	STZ !layer1_x,X	;screen scrolling distances
	REP #$20
	RTL

	REP #$20
	LDA.w #$0000-!Speed	;Screen is scrolling up or left
	BRA $05
	REP #$20
	LDA.w #$0000+!Speed	;Screen is scrolling down or right
	CLC
	ADC !layer1_x,X
	STA !layer1_x,X
	RTL

; Based on vanilla code at $82D961. We don't modify it in place because it is still
; used by fade-in (which is kept vanilla speed).
set_fadeout:
	LDA #$0006    ; double speed (vanilla: LDA #$000C)
	STA $7EC402	  ; Palette change denominator = Ch
	JSR $DA02  	  ; Advance gradual colour change of all palettes
	RTS

warnpc $82fa00
