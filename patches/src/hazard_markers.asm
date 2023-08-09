lorom

!bank_84_free_space_start = $84F580
!bank_84_free_space_end = $84F600
!bank_8f_free_space_start = $8FFE80
!bank_8f_free_space_end = $8FFF00

!hazard_tilemap_start = $E98280
!hazard_tilemap_size = #$0020

; landing site: testing using setup ASM
org $8F922B : dw spawn_right_hazard
org $8F9245 : dw spawn_right_hazard
org $8F925F : dw spawn_right_hazard
org $8F9279 : dw spawn_right_hazard

org $82E7A8
    jsl load_hazard_tiles

org $82E845
    jsl load_hazard_tilemap
    rep 3 : nop

org !bank_8f_free_space_start

spawn_right_hazard:
    JSL $88A7D8  ; vanilla setup ASM (scrolling sky)

    jsl $8483D7
    db $8F
    db $46
    dw right_hazard_plm
    rts

warnpc !bank_8f_free_space_end

org !bank_84_free_space_start


load_hazard_tiles:
    jsl $80B271  ; run hi-jacked instruction (decompress CRE tiles from $B98000 to VRAM $2800)

    LDA #$0080
    STA $2115  ; video port control
    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    lda #$00E9
    sta $4314  ; Set source bank to $E9

    LDA #$2A00
    STA $2116  ; VRAM (destination) address = $2A00
    lda #$8000 
    sta $4312  ; source address = $8000
    lda #$140
    sta $4315 ; transfer size = $140 bytes
    lda #$0002
    sta $420B  ; perform DMA transfer on channel 1

    LDA #$2B00
    STA $2116  ; VRAM (destination) address = $2B00
    lda #$8140 
    sta $4312  ; source address = $8140
    lda #$140
    sta $4315 ; transfer size = $140 bytes
    lda #$0002
    sta $420B  ; perform DMA transfer on channel 1

    rtl

load_hazard_tilemap:
    JSL $80B0FF  ; run hi-jacked instruction (Decompress CRE tile table to $7E:A000)
    dl $7EA000

    ldy !hazard_tilemap_size
    ldx #$0000
.loop:
    lda !hazard_tilemap_start, x
    sta $7EA700, x
    inx
    inx
    dey
    dey
    bne .loop

    rtl

right_hazard_plm:
    dw $B3D0, right_hazard_inst

right_hazard_inst:
    dw $0001, right_hazard_draw
    dw $86BC

right_hazard_draw:
    dw $8004, $00E0, $00E1, $08E1, $08E0, $0000

warnpc !bank_84_free_space_end