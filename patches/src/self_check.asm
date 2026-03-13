; Compute the ROM checksum in the background, using idle CPU time while waiting for NMI.
; If the value doesn't match the SNES header, then display an error screen and crash.
; This takes around 16 seconds if it finishes in the intro, title screen, or main menu.
; If the player jumps straight into the game, it can take somewhat longer, as there is
; less spare CPU available then.
;
; This patch is intended to help players notice up front if their ROM has been
; corrupted, e.g. by a bad SD card. It should help reduce cases of bug reports with
; strange crashes that we're not able to reproduce.
;
arch snes.cpu
lorom

!bank_8b_free_space_start = $8bf940
!bank_8b_free_space_end = $8bf960

!bank_80_free_space_start = $80BC37
!bank_80_free_space_end = $80C437

!chksum_fail_msg_start = $80B437
!chksum_fail_msg_end = $80BC37

!bank = $7e0336
!offset = $7e0337
!checksum = $7e0339

; hook the common boot section clear bank $7e routine and replace with one that won't clobber our checksum variables (already in use by this time)
;$80:8489 C2 30       REP #$30 <- starting here
;$80:84B1 E2 30       SEP #$30
;$80:84B3 9C 00 42    STZ $4200 <-- jumping back to here (the new clear sram routine calls sep #$30 so it leaves the routine in that state negating 
;                     the need for it again at 84b1.. it 84b3 is overwritten by nop in stats.asm but code returns to the vanilla location incase stats.asm ever changes.


org $808489
  jmp clear_7e_safe
  nop #39
  
; Hook the wait-for-NMI idle loop:

;$80:8340 8D B4 05    STA $05B4  [$7E:05B4]  ;} NMI request flag = 1
;$80:8343 AD B4 05    LDA $05B4  [$7E:05B4]  ;\
;$80:8346 D0 FB       BNE $FB    [$8343]     ;} Wait until NMI request acknowledged

org $808340
    jmp calc_checksum  ; hook to use spare CPU to compute checksum, while waiting for NMI
nmi_wait:
    lda $05b4          ; regular idle loop to wait for NMI (if already done computing checksum)
    bne nmi_wait
nmi_done:

org !bank_80_free_space_start

calc_checksum:
    sta $05B4           ; run hi-jacked instruction (NMI request flag = 1)
    lda !bank
    bne .do_checksum    ; non-zero = still computing checksum
    jmp nmi_wait        ; already done, return to vanilla NMI wait loop
    
.do_checksum
    phx
    phy
    php
    phb
    rep #$30            ; 16-bit X
    lda !offset
    tax
    sep #$20            ; 8-bit A
    lda !bank
    pha
    plb                 ; set DB to current bank
    lda !checksum+1
    tay
    lda !checksum
    clc
    pha
    
.chksum_loop:
    pla

macro add_byte(i)
    adc $0000+<i>,x
    bcc .no_carry<i>
    iny
    clc
.no_carry<i>:
endmacro

for i = 0..32
    %add_byte(!i)
endfor

    ; Increase X by 32:
    pha
    rep #$20
    txa
    adc #$0020
    tax
    sep #$20

    beq .new_bank

.same_bank:
    lda $8005b4
    beq .interrupted
    jmp .chksum_loop

.interrupted:
    ; NMI has finished, so save the current checksum state and return:
    pla
    sta !checksum
    tya
    sta !checksum+1
    rep #$20
    txa
    sta !offset             ; save offset
    plb
    plp
    ply
    plx
    jmp nmi_done

.new_bank:
    lda !bank
    inc
    sta !bank
    beq .done
    pha
    plb                     ; DB++
    ldx #$8000
    clc
    jmp .same_bank

.done:
    pla
    plb
    plp
    cmp $ffde
    bne .chkfail
    cpy $ffdf
    bne .chkfail
    ply
    plx
    jmp nmi_wait
        
.chkfail										; checksum doesnt match whats stored in ROM.. display error screen.
    stz $2140
    jsr $875d
    jsr $8792
    jsl $808b1a
    jsl $80896e
    lda #$8f
    sta $51
    sta $2100
    stz $4200
    stz $2116
    stz $2117
    lda #$80
    sta $2115
    jsl $8091a9
    db $01,$01,$18
    dw $8000
    db $8e
    dw $4000
    lda #$02
    sta $420b
    stz $2116
    lda #$40
    sta $2117
    lda #$80
    sta $2115
    jsl $8091a9
    db $01,$01,$18
    dw $b437
    db $80
    dw $1000
    lda #$02
    sta $420b
    stz $2121
    jsl $8091a9
    db $01,$00,$22
    dw $e400
    db $8e
    dw $0200
    lda #$02
    sta $420b
    stz $2131
    stz $212d
    lda #$01
    sta $212c
    lda #$0f
    sta $2100
    stz $210b
    lda #$40
    sta $2107
.crash
    bra .crash

clear_7e_safe: ;replaces the stock clear bank 7e unrolled stz routine on boot but does not clear the checksum locations.
;               dma clear (relies on rom header 80:0002 being vanilla 0000)
    sep #$30
    lda #$08
    sta $4300 
    lda #$80
    sta $4301 
    lda #$02  
    sta $4302   
    lda #$80
    sta $4303   
    lda #$80
    sta $4304
    stz $2181               
    stz $2182               
    stz $2183 
    lda #$35
    sta $4305
    lda #$03
    sta $4306
    lda #$01
    sta $420B
    lda #$40
    sta $2181               
    lda #$03
    sta $2182               
    stz $2183
    lda #$C0
    sta $4305 
    lda #$FC
    sta $4306             
    lda #$01
    sta $420B
    jmp $84B3 ; return to next instruction in common boot sequence.

assert pc() <= !bank_80_free_space_end

table "tables/menu.tbl",RTL
org !chksum_fail_msg_start ; replace the stock SRAM error with our own .

dw "                                "
dw "                                "
dw "                                "
dw "       SELF CHECK FAIL          "
dw "                                "
dw "   ROM CORRUPTION DETECTED      "
dw "                                "
dw "                                "
dw "                                "
dw "    VISIT MAPRANDO.COM FOR      "
dw "                                "
dw "    MORE INFORMATION OR JOIN    "
dw "                                "
dw "    THE DISCORD FOR HELP.       "
dw "                                "
dw " ------------------------------ "
dw "                                "
dw " COMMON CAUSES FOR THIS ERROR:  "
dw "                                "
dw "                                "
dw " - FAILING SD-CARD              "
dw "                                "
dw " - 3RD PARTY ROM PATCHING TOOLS "
dw "   NOT FIXING ROM CHECKSUM      "
dw "                                "
dw " - SD2SNES:FXPAK IN-GAME HOOKS  "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "

assert pc() <= !chksum_fail_msg_end

org $8b9155 ;$8B:9155 9C 90 05    STZ $0590  [$7E:0590]  ; OAM stack pointer = 0
  jsr init_chksum_variables

org !bank_8b_free_space_start
init_chksum_variables:
  stz $0590   ; hijacked instruction
  lda #$0080
  sta !bank
  xba
  sta !offset
  lda #$0000
  sta !checksum
  rts

assert pc() <= !bank_8b_free_space_end
