; Removes a spikesuit state from samus 

arch snes.cpu
lorom

!any_bank_free_space_start = $80E1B0
!any_bank_free_space_end = $80E2A0

org $90D4BC ; hook end of shinespark crash
    jsl check_ss
	nop
	nop

org !any_bank_free_space_start
check_ss:
	LDA $0ACC  		; Samus palette type normal? [regular shinecharge]
	BNE .skip
	LDA $0A68  		; special timer non zero? [can spark]
	BEQ .skip
	LDA #$0000
	STA $0A68 		; goodbye spikesuit
	LDA #$0045		; msg ID
	JSL $85B000 
	.skip:
	LDA #$0002
	STA $0A32
	STZ $0DEC
	RTL
	
assert pc() <= !any_bank_free_space_end
