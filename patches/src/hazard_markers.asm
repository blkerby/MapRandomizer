lorom

!bank_84_free_space_start = $84F580   ; must match address in patch.rs
!bank_84_free_space_end = $84F680
!bank_8f_free_space_start = $8FFE80
!bank_8f_free_space_end = $8FFF00

!bank_b5_free_space_start = $B5F700
!bank_b5_free_space_end = $B5F800



!hazard_tilemap_start = $E98280
!hazard_tilemap_size = #$0020


org $82E7BB
    jsl load_hazard_tiles

org $82E845
    jsl load_hazard_tilemap_initial_hook
    rep 3 : nop

org $82E42E
    jsl reload_hazard_tiles
    rep 3 : nop

; hook extra setup ASM to run right before normal setup ASM
; (careful: escape.asm hijacks the instruction after this)
org $8FE893
    jsr run_extra_setup_asm_wrapper

org !bank_b5_free_space_start

run_extra_setup_asm:
    ; get extra setup ASM pointer (vanilla-unused pointer in room state), to run in bank B5
    LDX $07BB
    LDA $0010,x
    beq .skip
    sta $00         ; write setup ASM pointer temporarily to direct page $00, so we can jump to it with JSR.
    ldx #$0000
    jsr ($0000,x)
.skip:
    ; run hi-jacked instructions
    LDX $07BB
    LDA $0018,x
    rtl

warnpc !bank_b5_free_space_end

org !bank_8f_free_space_start

run_extra_setup_asm_wrapper:
    jsl run_extra_setup_asm
    rts

warnpc !bank_8f_free_space_end


org !bank_84_free_space_start

; These PLMs definitions must go here first, as they are referenced in patch.rs
right_hazard_transition_plm:
    dw $B3D0, right_hazard_transition_inst

down_hazard_plm:
    dw $B3D0, down_hazard_inst

down_hazard_transition_plm:
    dw $B3D0, down_hazard_transition_inst

load_hazard_tiles:
    jsl $80B271  ; run hi-jacked instruction (Decompress [tileset tiles pointer] to VRAM $0000)

    LDA #$0080
    STA $2115  ; video port control
    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    lda #$00E9
    sta $4314  ; Set source bank to $E9

    LDA #$2600
    STA $2116  ; VRAM (destination) address = $2600
    lda #$8000 
    sta $4312  ; source address = $8000
    lda #$0100
    sta $4315 ; transfer size = $100 bytes
    lda #$0002
    sta $420B  ; perform DMA transfer on channel 1

    rtl

load_hazard_tilemap_initial_hook:
    JSL $80B0FF  ; run hi-jacked instruction (Decompress CRE tile table to $7E:A000)
    dl $7EA000
; falls through to below:
load_hazard_tilemap:
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

reload_hazard_tiles:
    ; run-hijacked instruction (decompress room tiles)
    jsl $80B0FF
    dl $7E2000

    ; Copy hazard tiles from $E98000-$E98100 to $7E6C00
    ldx #$00FE
-
    lda $E98000,x
    sta $7E6C00,x
    dex
    dex
    bpl -

    ; Copy hazard tilemap (definition of 16 x 16 tiles from 8 x 8 tiles)
    jsl load_hazard_tilemap

    rtl

right_hazard_transition_inst:
    dw $0001, right_hazard_transition_draw
    dw $86BC

right_hazard_transition_draw:
    dw $8004, $90E0, $90E1, $98E1, $98E0, $0000

down_hazard_inst:
    dw $0001, down_hazard_draw
    dw $86BC

down_hazard_draw:
    dw $0004, $00E2, $00E3, $04E3, $04E2, $0000

down_hazard_transition_inst:
    dw $0001, down_hazard_transition_draw
    dw $86BC

down_hazard_transition_draw:
    dw $0004, $90E2, $90E3, $94E3, $94E2, $0000

warnpc !bank_84_free_space_end