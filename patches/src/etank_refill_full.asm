lorom

org $848972
	jsr $F700

org $84F700
	sta $09C2 ;} Original instruction
	lda $09D4 ;\
	sta $09D6 ;} Refill reserves
	rts
