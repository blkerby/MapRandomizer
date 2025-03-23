lorom

!bank_84_free_space_start = $84F800   ; must match address in patch.rs
!bank_84_free_space_end = $84F900
!bank_8f_free_space_start = $8FFE80
!bank_8f_free_space_end = $8FFF00

!bank_b8_free_space_start = $B88000
!bank_b8_free_space_end = $B88100

!hazard_tilemap_start = $E98280
!hazard_tilemap_size = #$0020

; hook initial load and unpause
org $82E7BF
    jsl load_hazard_tiles : nop

org $82E845
    jsl load_hazard_tilemap_initial_hook
    rep 3 : nop

; hook door transition
org $82E42E
    jsl reload_hazard_tiles
    rep 3 : nop

; hook extra setup ASM to run right before normal setup ASM
; (careful: escape.asm hijacks the instruction after this)
org $8FE893
    jsr run_extra_setup_asm_wrapper

org !bank_b8_free_space_start

run_extra_setup_asm:
    ; get extra setup ASM pointer to run in bank B5 (using pointer in room state almost completely unused by vanilla, only for X-ray override in BT Room in escape)
    LDX $07BB
    LDA $0010,x
    beq .skip
    sta $1F68         ; write setup ASM pointer temporarily to $1F68, so we can jump to it with JSR. (Is there a less awkward way to do this?)
    ldx #$0000
    jsr ($1F68, x)

.skip:
    ; run hi-jacked instructions
    LDX $07BB
    LDA $0018,x
    rtl

warnpc !bank_b8_free_space_end

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

left_hazard_transition_plm:
    dw $B3D0, left_hazard_transition_inst

load_hazard_tiles:
    ; Load hazard tiles
    LDA #$0080
    STA $2115  ; video port control
    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    lda #$00E9
    sta $4314  ; Set source bank to $E9
    LDA #$2780
    STA $2116  ; VRAM (destination) address = $2780
    lda #$8000 
    sta $4312  ; source address = $8000
    lda #$0100
    sta $4315 ; transfer size = $100 bytes
    lda #$0002
    sta $420B  ; perform DMA transfer on channel 1

    ; Load beam door tiles
    lda #$00EA
    sta $4314  ; Set source bank to $EA
    lda #$2700
    sta $2116  ; VRAM (destination) address = $2700
    lda $1F78
    clc
    adc #$0020
    sta $4312  ; source address = [$1F78] + $0020
    lda #$0100
    sta $4315 ; transfer size = $100 bytes
    lda #$0002
    sta $420B  ; perform DMA transfer on channel 1    

    lda $7c7   ; replaced code
    sta $48

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

    ; Copy hazard tiles from $E98000-$E98100 to $7E6F00
    ldx #$00FE
-
    lda $E98000,x
    sta $7E6F00,x
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

left_hazard_transition_inst:
    dw $0001, left_hazard_transition_draw
    dw $86BC

left_hazard_transition_draw:
    dw $8004, $94E0, $94E1, $9CE1, $9CE0, $0000

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