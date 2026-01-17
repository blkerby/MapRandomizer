; Removes a spikesuit state from samus 

arch snes.cpu
lorom

!bank_90_free_space_start = $90FC20
!bank_90_free_space_end = $90FC40

org $90D4B9 ; hook end of shinespark crash
    jsl check_ss
	nop
	nop

org !bank_90_free_space_start
check_ss:
	LDA $0ACC  		; Samus palette type normal? [regular shinecharge]
	BNE .skip
	LDA $0A68  		; special timer non zero? [can spark]
	BEQ .skip
	STZ $0A68 		; goodbye spikesuit
	LDA #$0019
	JSL $8090CB		; play a sound effect
	.skip:

	; Run hi-jacked instructions:
	LDA #$0002
	STA $0A32
	RTL
	
assert pc() <= !bank_90_free_space_end