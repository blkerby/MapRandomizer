; Based on https://github.com/tewtal/sm_practice_hack/blob/723e3a6d18a3f66903cc067cd923e0a6b28aebf5/src/cutscenes.asm#L29

arch snes.cpu

!bank_80_freespace_start = $80D100
!bank_80_freespace_end = $80D140
!bank_82_freespace_start = $82F900
!bank_82_freespace_end = $82FA00


!Speed = 2
!layer1_x = $0911
!layer1_y = $0915


org $80AE5C
    JSR door_transition

org !bank_80_freespace_start
; Run the door transition twice per frame.
door_transition:
    PHX
    JSR ($AE76,x)
    PLX
    BCC .slow
    RTS             ; If the door transition is done, don't run it again.
.slow
    JMP ($AE76,x)

warnpc !bank_80_freespace_end

;;; Door centering speed by Kazuto:
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

;org $82E2DB
;	JSR set_fadeout

org !bank_82_freespace_start
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

warnpc !bank_82_freespace_end