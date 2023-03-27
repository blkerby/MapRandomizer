org $90ECB6
set_suit_index:
    PHX
    LDX #$0000
    LDA $09A2
    BIT #$0020
    BEQ .no_gravity
    INX
    INX
    INX
    INX
.no_gravity
    BIT #$0001
    BEQ .no_varia
    INX
    INX
.no_varia
    STX $0A74
    PLX
    RTS
warnpc $90ECD5

org $91DEBA
    php
    phb
    phk
    plb
    sep #$30
    lda $09a2
    ldy #$00
    bit #$20              ; has gravity?
    beq +
    iny #4
    +
    bit #$01              ; has varia?
    beq +
    iny #2
+
    rep #$30
    ldx.w pal_data,y
    jsr $DD5B
    plb
    plp
    rtl
    
pal_data:
    dw $9400, $9520, $FF00, $9800

org $91DEE6
    php
    phb
    phk
    plb
    sep #$30
    lda $09a2
    ldy #$00
    bit #$20              ; has gravity?
    beq +
    iny #4
    +
    bit #$01              ; has varia?
    beq +
    iny #2
+
    rep #$30
    ldx.w pal_data,y
    jsr $DDD7
    plb
    plp
    rtl

org $91DE6A
    jsl $91DEBA
    bra +

org $91DE8D
    +

org $91D71B
    dw pal_data

