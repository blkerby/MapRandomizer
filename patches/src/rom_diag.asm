;-- SM ROM DIAGNOSIS (FULL BYTE COUNT CHECKSUM VERIFICATION)
;-- nn_357

arch snes.cpu
lorom

!bank_80_free_space_start = $8085F6 ; this is where the sram/region check used to live.
!bank_80_free_space_end = $80875B   ; and this is where it ended.
!checksum = $7E1F00
!checksum_hi = $7E1F01
!checksum_compl = $7E1F04
!checksum_compl_hi = $7E1F05
!toggle = $7E1F03
!NextBank = $7E1F02    ; current bank
;!NextWord = $7E1F04		 ; next word
!ROM_BANKS      = #$80       ; 4 MB / 32 KB
!BANK_HEADER    = #$80
!BYTES_PER_BANK = #$8000

			; Remove the original call to region/sram check, Maprando bypasses this anyway.
org $80855F		; $80:855F 20 F6 85    JSR $85F6  [$80:85F6]  ; NTSC/PAL and SRAM mapping check
	NOP
	NOP
	NOP

			; Let's not go to the main game loop just yet. Check to see if the diagnostic hot-keys are being held.
org $80856E		; $80:856E 5C 3D 89 82 JML $82893D[$82:893D]  ; Go to main game loop
	JML diag		
	
	
org !bank_80_free_space_start
diag:
	LDA #$0000					;
	STA !checksum					; THIS LDA and STA x 3 can be removed for space if needed (not 100% safe) as the WRAM is initialized immediatly before.
	STA !NextBank					;
	STA !checksum_compl		
	LDA $4218 					; see what keys are being pressed
	CMP #$3800 					; is it START + SELECT + UP ?
	BNE .diag_skip
	;STZ $4200					; disable NMI
	;STZ $420C					; disable IRQ / TIMERS etc - test
	LDY #$0000
	LDX !BYTES_PER_BANK
	SEP #$20					; 8 bit accumulator
	LDA #$00
	STA !NextBank
	LDA #$03      					; Music ID (item room / computer sounding thing)
	STA $2140  
	JSR screencolors
		
.diagloop:
	LDA !NextBank
	CMP !ROM_BANKS
	BCS .finish_bank				; all banks done
	LDA !NextBank					; map bank to $80+
	ORA #$80
	PHA
	PLB 	
	LDY #$0000
	LDX !BYTES_PER_BANK
	
		
.loop_words:
	LDA $8000,Y
	PHA
	LDA !NextBank
	CMP #$00
	BNE .do_add
	CPY #$7FDC
	BEQ .checkbyte
	CPY #$7FDD
	BEQ .checkbyte
	CPY #$7FDE
	BEQ .complibyte
	CPY #$7FDF
	BEQ .complibyte
		
.do_add:
	PLA
	CLC
	ADC !checksum
	STA !checksum
	BCC .conti
	LDA !checksum_hi
	INC
	STA !checksum_hi

.conti:
	INY
	DEX
	BNE .loop_words
	LDA !NextBank
	INC
	STA !NextBank
	JSR toggle_screen
	BRA .diagloop

.finish_bank:
	BRA .dochecksumxor

.diag_skip:
	JML $82893D		;	go to main game loop

.checkbyte:
	PLA
	LDA #$FF
	PHA
	BRA .do_add
.complibyte:
	PLA
	LDA #$00
	PHA
	BRA .do_add
		
.dochecksumxor:
	LDA !checksum
	EOR #$FF
	STA !checksum_compl
	LDA !checksum_hi
	EOR #$FF
	STA !checksum_compl_hi
		
.final_check:
	LDA #$80
	PHA
	PLB
	LDA !checksum_compl
	CMP $FFDC 
	BNE .chkfail
	LDA !checksum_compl_hi
	CMP $FFDD
	BNE .chkfail
	LDA !checksum
	CMP $FFDE
	BNE .chkfail
	LDA !checksum_hi
	CMP $FFDF
	BNE .chkfail
	LDA #$02      ; Music ID (fanfare )
	STA $2140
	LDA #$80
	STA $2100
	STZ $2121
	LDA #$E0
	STA $2122
	LDA #$03
	STA $2122
	LDA #$0F
	STA $2100
.passloop:
	REP #$30
	LDA $4218 		; see what keys are being pressed
	CMP #$1000		; checking for START
	BNE .passloop
	JML $80841C		; reboot game, should already be in correct dbr...
		
.chkfail:			; checksum doesnt match whats stored in ROM.. display a red screen and crash.
	LDA #$80
	STA $2100
	STZ $2121
	LDA #$1F
	STA $2122
	LDA #$00
	STA $2122
	LDA #$0F
	STA $2100
.infi_loop:
	BRA .infi_loop
		
screencolors:		; setup the blue screen
	LDA #$80
	STA $2100
	STZ $212C
	STZ $212D
	STZ $2130
	STZ $2131
	STZ $2105
	STZ $2101
	STZ $2121
	LDA #$00
	STA $2122
	LDA #$7C
	STA $2122
	LDA #$0F
	STA $2100
	RTS
		
toggle_screen:
.wait_vblank:
	LDA !toggle
	EOR #$01
	STA !toggle
	BEQ .low
.high:
	LDA #$0F
	STA $802100
	RTS
.low:
	LDA #$08
	STA $802100
	RTS
		
print pc
assert pc() <= !bank_80_free_space_end