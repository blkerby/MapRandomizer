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

!bank_80_free_space_start = $80BC37
!bank_80_free_space_end = $80C437
!sram_msg_end = $80BC37

!bank = $7ffa02
!offset = $7ffa03
!checksum = $7ffa05


; bootloop burns 4 frames during bootup, our init code fits exactly here and avoids a jump.

;$80:8438 E2 30       SEP #$30
;$80:843A A2 04       LDX #$04               ;\
;                                            ;|
;$80:843C AD 12 42    LDA $4212              ;|
;$80:843F 10 FB       BPL $FB    [$843C]     ;|
;                                            ;} Wait the remainder of this frame and 3 more frames
;$80:8441 AD 12 42    LDA $4212              ;|
;$80:8444 30 FB       BMI $FB    [$8441]     ;|
;$80:8446 CA          DEX                    ;|
;$80:8447 D0 F3       BNE $F3    [$843C]     ;/
;$80:8449 C2 30       REP #$30
;$80:844B A2 FE 1F    LDX #$1FFE  

org $808438
    lda #$0080
    sta !bank
    xba
    sta !offset
    lda #$0000
    sta !checksum

assert pc() <= $80844B ; lets not overwrite the next instruction.

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