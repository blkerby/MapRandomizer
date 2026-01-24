;-- SM ROM DIAGNOSIS (FULL BYTE COUNT CHECKSUM VERIFICATION)


arch snes.cpu
lorom

!bank_80_free_space_start = $8085F6 ; this is where the sram/region check used to live.
!bank_80_free_space_end = $80875B   ; and this is where it ended.

!checksum = $7E1F31
!bank = $1F33    ; current bank
!current_offset = $1F37
!initialized_checksum = $1F35


;!ROM_BANKS      = #$80       ; 4 MB / 32 KB
;!BYTES_START_ADDR = #$8000

							
org $80855F		; $80:855F 20 F6 85    JSR $85F6  [$80:85F6]  ; NTSC/PAL and SRAM mapping check
		JSR init_chksum
		
org $808343
		JSR bg_checksum
		NOP
		NOP
		
org !bank_80_free_space_start
bg_checksum:
		
	
		LDA !initialized_checksum
		CMP #$69
		BNE .alldoneornotstarted
		LDA !bank
		CMP #$80
		BEQ .alldoneornotstarted
		
		
		
		ORA #$80						; map bank to $80+
		PHA
		REP #$30						; 16 bit EVERYTHING;;
		LDX !current_offset
		LDA $0998
		AND #$00FF
		CMP #$0007
		BCS .ingame
		LDY #$0040
		BRA .title
.ingame:
		LDY #$0002
.title:
		PLB 								; set the dbr to whatever bank we are reading
		BRA .diagloop
		
.alldoneornotstarted:
		SEP #$30
		LDA $05B4		
		BNE	.alldoneornotstarted
		RTS
		
.diagloop:				;	 4 bytes at a time x 4 loops (testing)
		LDA $7E1F33
		CMP #$0080
		BEQ .alldoneornotstarted
		
		CMP #$0000
		BNE .normal
		CPX #$FFDC
		BNE .normal
		LDA #$FFFF
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA #$FFFF
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA #$0000
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA #$0000
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		BRA .finished_loop
		
.normal:
		
		LDA $0000, X
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA $0000, X
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA $0000, X
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		LDA $0000, X
		AND #$00FF
		CLC
		ADC !checksum
		STA !checksum
		INX
		CPX #$0000
		BEQ .wraparound
		DEY
		BNE .normal
		
.finished_loop	
		PHK 
		PLB
		LDA $05B4							; hijacked instruction (wait for NMI acknowledge)
		AND #$00FF
		BNE	.continiue					; no NMI yet, read more.
		STX	!current_offset		;	park our current offset.
		SEP #$30							; put cpu back into 8 bit mode everything.
		RTS										; return.
.continiue:
		
		LDA $0998
		AND #$00FF
		CMP #$0007
		BCS .ingame2
		LDY #$0040
		JMP .notingame
.ingame2:
		LDY #$0002
.notingame:
		LDA !bank
		AND #$00FF
		ORA #$0080
		XBA
		PHA 
		PLB
		PLB
		JMP .diagloop
		
.wraparound:
		PHK 
		PLB
		LDX #$8000
		STX !current_offset
		LDA !bank
		AND #$00FF
		INC
		STA !bank
		BRA .finished_loop
		

init_chksum:
		SEP #$20		      	    ; A 8-bit
    REP #$10    			      ; X/Y 16-bit
		LDA #$00
		STA !bank
		LDA #$69
		STA !initialized_checksum
		LDX #$8000
		STX !current_offset			; initialize our variable.
		REP #$30
		RTS
		
print pc
assert pc() <= !bank_80_free_space_end