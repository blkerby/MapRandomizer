arch snes.cpu
lorom

!bank_80_free_space_start = $8085F6 ; this is where the sram/region check used to live.
!bank_80_free_space_end = $80875B   ; and this is where it ended.
!sram_msg_end = $80BC37

!bank = $9c
!offset = $9d
!checksum = $9f


; $80:855F 20 F6 85    JSR $85F6  [$80:85F6]  ; NTSC/PAL and SRAM mapping check
                            
org $80855F						; original call to the SRAM routine, we can use this to setup our RAM variables.
    JSR hook_init 
        
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
hook_init:
    lda #$0080
    sta !bank
    xba
    sta !offset
    stz !checksum
    rts

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
    rep #$10            ; 16-bit X
    ldx !offset
    lda !bank
    pha
    plb                 ; set DB to current bank
    lda !checksum
    ldy !checksum+1
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

    %add_byte(0)
    %add_byte(1)
    %add_byte(2)
    %add_byte(3)
    %add_byte(4)
    %add_byte(5)
    %add_byte(6)
    %add_byte(7)
    %add_byte(8)
    %add_byte(9)
    %add_byte(10)
    %add_byte(11)
    %add_byte(12)
    %add_byte(13)
    %add_byte(14)
    %add_byte(15)

    pha
    rep #$20
    txa
    adc #$0010
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
    sty !checksum+1
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
        
.chkfail										; checksum doesnt match whats stored in ROM.. display a red screen and crash.
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

print pc
assert pc() <= !bank_80_free_space_end

table "tables/menu.tbl",RTL
org $80B437 ; replace the stock SRAM error with our own .

dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "       SELF CHECK FAIL          "
dw "                                "
dw "     CORRUPT ROM DETECTED       "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "    VISIT MAPRANDO.COM FOR      "
dw "                                "
dw "    MORE INFORMATION OR JOIN    "
dw "                                "
dw "    THE DISCORD FOR HELP.       "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "       (COMMON CAUSE IS A       "
dw "                                "
dw "        FAILING SD-CARD)        "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "
dw "                                "

assert pc() <= !sram_msg_end