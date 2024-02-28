lorom

!bank_80_free_space_start = $80E100
!bank_80_free_space_end = $80E180
!main_palette_table_addr = $80DD00

; Hook "load state header" to modify palette address
org $82DF23
    jsl load_palette
    nop : nop

org !bank_80_free_space_start
GetTilesetIndex:
    PHX
    LDX $07BB
    LDA $8F0003,X
    PLX
    AND #$00FF
    RTS

load_palette:
    phx
    phb
    phk
    plb

    ; $14 <- [main_palette_table_addr + map_area * 2]
    lda $1F5B  ; map area
    asl
    tax
    lda !main_palette_table_addr, x
    sta $14

    ; A <- tileset index * 3
    jsr GetTilesetIndex
    sta $12
    asl
    clc
    adc $12

    ; x <- [main_palette_table_addr + map_area * 2] + tileset index * 3
    adc $14
    tax

    ; store palette pointer in 3 bytes at $07C6
    lda $0001, x
    sta $07C7
    lda $0000, x
    sta $07C6

    plb
    plx
    rtl

warnpc !bank_80_free_space_end
