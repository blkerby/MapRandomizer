arch snes.cpu
lorom

!bank_80_free_space_start = $8085F6 ; this is where the sram/region check used to live.
!bank_80_free_space_end = $80875B   ; and this is where it ended.

; $1f89: bank
; $1f8a: offset
; $1f8c: checksum


; $80:855F 20 F6 85    JSR $85F6  [$80:85F6]  ; NTSC/PAL and SRAM mapping check
							
org $80855F						; original call to the SRAM routine, we can use this to setup our RAM variables.
		JSR hook_init 
		
;$80:8343 AD B4 05    LDA $05B4  [$7E:05B4]  ;\
;$80:8346 D0 FB       BNE $FB    [$8343]     ;} Wait until NMI request acknowledged
		
org $808343
    jmp calc_checksum : nop : nop
post_hook:
		
org !bank_80_free_space_start
hook_init:
    lda #$0080
    sta $1f89
    xba
    sta $1f8a
    stz $1f8c
    rts

calc_checksum:
    lda $1f89           ; curr bank
    bne .do_checksum    ; non-zero = processing
.nmi_wait
    lda $5b4
    bne .nmi_wait
    jmp post_hook
    
.do_checksum
    php
    phb
    rep #$10            ; 16-bit X
    ldx $1f8a           ; curr offset
    lda $1f89           ; bank
    pha
    plb                 ; set DB to current bank
    
.chksum_loop
    lda $0000,x
    clc
    adc $801f8c
    sta $801f8c
    bcc .no_carry
    lda $801f8d
    inc
    sta $801f8d
.no_carry
    inx
    bne .same_bank
    lda $801f89
    inc
    beq .done
    sta $801f89
    pha
    plb                     ; DB++
    lda #$00
    sta $801f8a
    lda #$80
    sta $801f8b
    ldx #$8000
    
.same_bank
    lda $8005b4
    bne .chksum_loop
    rep #$20
    txa
    sta $801f8a             ; save offset
    plb
    plp
    jmp post_hook

.done
    sta $801f89             ; 00 (done)
		plb
    plp
		lda $1f8c
		cmp $ffde
		bne .chkfail
		lda $1f8d
		cmp $ffdf
		bne .chkfail
    bra .nmi_wait
		
.chkfail										; checksum doesnt match whats stored in ROM.. display a red screen and crash.
		stz $2140
		stz $4200			
		stz $420C
		stz $212C
		stz $212D
		stz $2130
		stz $2131
		stz	$2105
		stz $2101
		lda #$80
		sta $2100
		stz $2121
		lda #$1F
		sta $2122
		lda #$00
		sta $2122
		lda #$0F
		sta $2100
.infi_loop:
		bra .infi_loop
		
		
print pc
assert pc() <= !bank_80_free_space_end