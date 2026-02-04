!bank_85_free_space_start = $85AE13
!bank_85_free_space_end = $85B000

!arrow_mode = $7EF597

; Hook runs when the dpad is pressed in the SUPPLY section
; (Note: disableable_etanks also hooks at a later point in this routine, at ACEF (BRANCH_UP).)
org $82AC8E
    jml hook_tanks_dpad_response
    nop
    
; Load equipment menu
org $8291B8
    lda #$0000
    jsr hook_init_arrow_mode

org $82AD08
    ; There's a PLP / RTS here for when we need to escape hooked $82 routines that had to JML out to $85
fake_rtl:

; Hook code that prepares the reserve tank delay sound counter
org $82AF5E
    jml hook_reserve_delay_counter_init

; Hook code that performs the reserve tank refill
org $82AF81
    jml hook_reserve_tank_refilling

; Jump table for the equipment tanks category (A button handler)
org $82AC7C
    jsr (hook_tanks_items, x)

org $82AC87
    ; Having hijacked the jump table that's normally here, we can repurpose it for a little bit of code...
hook_tanks_arrow_trampoline:
    jml hook_tanks_arrow

org $82AD0A
    php
    rep #$30
    lda $0755
    bit #$00FF
    bne .off
    xba
    lsr a
    bcs .solid
    jsr $AD29
    bra .done
.solid
    jsr $ADDD
    bra .done
.off
    jsr $ADEF
.done
    plp
    rts
warnpc $82AD29


warnpc $82AF6B
org $82AF6B
process_reserve_counter:


org $82FF20
hook_tanks_items:
    dw $AE8B, $AF4F, hook_tanks_arrow_trampoline

extern_disable_arrow_glow:
    jsr $AE46
    rtl

warnpc $82FF30

org $82F9F6
hook_init_arrow_mode:
    ; A = 0
    sta !arrow_mode
    inc a
    sta $0763 ;Hijacked (pause menu mode = equipment screen)
    rts
warnpc $82FA00

org !bank_85_free_space_start

!ram_bg1_tilemap_arrow_top = $7E3902
!ram_bg1_tilemap_arrow_end = $7E3B04
!arrow_top_normal_tile = $3D4C ; Vanilla tile (upward arrowhead)
!arrow_end_normal_tile = $3D6F ; Vanilla tile (short horizontal end)
!arrow_top_reversed_tile = $3D5C ; Vanilla tile (vertical line)
!arrow_end_reversed_tile = $3D4B ; Custom tile (rightward arrowhead)

hook_tanks_arrow:
    php
    lda $8F
    bit #$0080
    beq .no
    lda !arrow_mode
    eor #$0001
    sta !arrow_mode
    asl
    tax
    lda.l arrow_top_tile, x
    sta !ram_bg1_tilemap_arrow_top
    lda.l arrow_end_tile, x
    sta !ram_bg1_tilemap_arrow_end
    stz $0757
.no
    jml fake_rtl

arrow_top_tile:
    dw $3D4C, $3D5C

arrow_end_tile:
    dw $3D6F, $3D4B

hook_tanks_dpad_response:
    ; Arrived here by JML from an $82 JSR - MUST JML back to an $82 PLP/RTS!
    ;Hijacked instructions
    lda $0755
    sta $12
    
    lda $8F
    bit #$0200  ; P1 D-Pad Left
    beq .not_left

    ; Move to the arrow
    lda $0755
    and #$00FF
    ora #$0200
    sta $0755
    bra .moved

.not_left

    lda $12
    xba
    and #$00FF
    cmp #$0002  ; Arrow selected
    bne .not_arrow
    
    lda $8F
    bit #$0100  ; P1 D-Pad Right
    beq .not_right
    
    ; Move back to the Mode select
    stz $0755
.moved
    lda #$0037
    jsl $809049 ; Play sound: moved cursor
    
.not_right
    jml fake_rtl ; Skip the rest of the vanilla routine
    
.not_arrow
    jml $82AC93 ; Return to vanilla code

hook_reserve_delay_counter_init:
    lda !arrow_mode
    beq .vanilla
    lda $09c2   ; Reserve tank delay sound counter = [Samus health] - 1, rounded up to nearest 8
    dec a
    clc
    adc #$0007
    and #$fff8
    sta $0757
    jml $82AF6B ; Return to vanilla code at handling the reserve tank delay sound counter
.vanilla
    lda $09d6   ; Hijacked code
    clc
    jml $82AF62 ; Return to vanilla code

hook_reserve_tank_refilling:
    lda !arrow_mode
    beq .vanilla
    
    ; Samus health -= 1, unless this would be 0
    lda $09C2
    dec a
    beq .stop_etanks_empty
    sta $09C2
    lda $09D6
    inc a
    cmp $09D4
    bpl .stop_dump_etanks
    sta $09D6
    bra .done
    
.stop_dump_etanks
    ; Reserves hit full, vent regular energy
    lda $09D4
    sta $09D6
    lda #$0001
    sta $09C2
    jsl extern_disable_arrow_glow
    stz $0755

.stop_etanks_empty
    stz $0757

.done
    jml fake_rtl

.vanilla
    lda $09C2   ; Hijacked code
    clc
    jml $82AF85 ; Return to vanilla function

warnpc !bank_85_free_space_end

; There's a free spot right here in the vanilla pause BG1 tiles
org $B6A960
    db $00, $20, $20, $30, $30, $38, $38, $FC, $FC, $FC, $38, $38, $30, $30, $20, $20
    db $70, $00, $58, $20, $CC, $30, $C6, $38, $02, $FC, $C4, $38, $48, $30, $50, $20
