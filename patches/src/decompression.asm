; From https://github.com/theonlydude/RandomMetroidSolver/blob/master/patches/common/src/decompression.asm
;$30FF-$3265

;;; Decompression optimization by Kejardon, with fixes by PJBoy and Maddo

;Compression format: One byte(XXX YYYYY) or two byte (111 XXX YY-YYYYYYYY) headers
;XXX = instruction, YYYYYYYYYY = counter
lorom
org $80B0FF
	
	LDA $02, S
	STA $45
	LDA $01, S
	STA $44		;Address of target address for data into $44
	CLC
	ADC #$0003
	STA $01, S	;Skip target address for RTL
	LDY #$0001
	LDA [$44], Y
	STA $4C
	INY
	LDA [$44], Y
	STA $4D
;80B119 : Later JSL start if 4C (target address) is specified
	PHP
	PHB
	SEP #$20
	REP #$10	;This is only POSSIBLY useful from JSL 80B119...
	LDA $49
	PHA
	PLB
	STZ $50
	LDY #$0000
	BRA NextByte

End:
	PLB
	PLP
	RTL

NextByte:
	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+
	STA $4A
	CMP #$FF
	BEQ End
	CMP #$E0
	BCC ++
	ASL A
	ASL A
	ASL A
	AND #$E0
	PHA
	LDA $4A
	AND #$03
	XBA

	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+
	BRA +++

++

	AND #$E0
	PHA
	TDC
	LDA $4A
	AND #$1F

+++

	TAX
	INX
	PLA

	BMI Option4567

	BEQ Option0

	CMP #$20

	BEQ BRANCH_THETA

	CMP #$40

	BEQ BRANCH_IOTA

;X = 3: Store an ascending number (starting with the value of the next byte) Y times.
	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+

-

	STA [$4C], Y
	INC A
	INY
	DEX

	BNE -

	JMP NextByte

Option0:
-

	LDA ($47)	;X = 0: Directly copy Y bytes
	INC $47
	BNE +
	JSR IncrementBank2
+

	STA [$4C], Y
	INY
	DEX

	BNE -

	JMP NextByte

BRANCH_THETA:

	LDA ($47)	;X = 1: Copy the next byte Y times.
	INC $47
	BNE +
	JSR IncrementBank2
+

-

	STA [$4C], Y
	INY
	DEX

	BNE -

	JMP NextByte

BRANCH_IOTA:
	REP #$20 : TXA : LSR : TAX : SEP #$20 ; PJ: X /= 2 and set carry if X was odd

	LDA ($47)	;X = 2: Copy the next two bytes, one at a time, for the next Y bytes.
	INC $47
	BNE +
	JSR IncrementBank2
+

	XBA

	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+
	XBA

	INX ; Test if X = 0 without overwriting carry
	DEX
	BEQ ++  ; If X = 0 then skip the loop (otherwise X would underflow and the loop would run $10000 times)

	REP #$20

-

	STA [$4C], Y
	INY
	INY
	DEX
	BNE -

	SEP #$20

++
	BCC + : STA [$4C], Y : INY : + ; PJ: If carry was set, store that last byte

	JMP NextByte



Option4567:

	CMP #$C0

	AND #$20	;X = 4: Copy Y bytes starting from a given address in the decompressed data.
	STA $4F	;X = 5: Copy and invert (EOR #$FF) Y bytes starting from a given address in the decompressed data.

	BCS +++	;X = 6 or 7 branch

	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+
	XBA
	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+
	XBA
	REP #$21
	ADC $4C
	STY $44
	SEC
--
	SBC $44
	STA $44
	SEP #$20

;XBA : XBA : REP : ADC dp : STY dp : SEC : SBC dp : STA dp : SEP. 3+3+3+4+4+2+4+4+3 = 30
;STA dp : STA dp : REP : TYA : EOR : INC : ADC dp : CLC : ADC dp : STA dp : SEP. 3+3+3+2+3+2+4+2+4+4+3 = 33

;--
	LDA $4E
	BCS +
	DEC
+
	STA $46
+
	LDA $4F
	BNE +	;Inverted

-
	LDA [$44], Y
	STA [$4C], Y
	INY
	DEX

	BNE -
	JMP NextByte

+
-
	LDA [$44], Y
	EOR #$FF
	STA [$4C], Y
	INY
	DEX

	BNE -
	JMP NextByte


+++

	;X = 6: Copy Y bytes starting from a given number of bytes ago in the decompressed data.
	;X = 7: Copy and invert (EOR #$FF) Y bytes starting from a given number of bytes ago in the decompressed data.

	TDC

	LDA ($47)
	INC $47
	BNE +
	JSR IncrementBank2
+

	REP #$20
	STA $44
	LDA $4C
;	SEC	;I think I can get away without this :D
;	SBC $4A
;	STA $44
;	SEP #$20

	BRA --


IncrementBank2:
	INC $48
	BNE +
	PHA
	PHB
	PLA
	INC A
	PHA
	PLB
	LDA #$80
	STA $48
	PLA
+
	RTS

warnpc $80B266