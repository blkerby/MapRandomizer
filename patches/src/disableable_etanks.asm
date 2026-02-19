!bank_80_free_space_start = $80E3C0
!bank_80_free_space_end = $80E440

!bank_82_free_space_start = $82F830
!bank_82_free_space_end = $82F9E0

!bank_83_free_space_start = $83BB00
!bank_83_free_space_end = $83BC40

!current_etank_index = $12
!count_full_etanks = $14
!count_enabled_etanks = $16
!count_all_etanks = $18
!etank_hud_tile_offset = $1A


incsrc "constants.asm"

org $809698
    lda $9C
assert pc() == $80969A

org $809BEE
    jsr hook_draw_tanks

org !bank_80_free_space_start

hook_draw_tanks:
    lda !num_disabled_etanks
    beq .done
    sta $16
.loop:
    ldx $9CCE,y
    lda #$3C2F
    sta $7EC608,x
    iny
    iny
    dec $16
    bne .loop

.done:
    ; Make the entire energy area of the HUD not be priority so that we can draw sprites on it
    lda $0998
    cmp #$000f    ; Not pause menu?
    bne .dont

    ldx #$0012
-
    lda $7EC608,x
    and #$DFFF
    sta $7EC608,x
    lda $7EC648,x
    and #$DFFF
    sta $7EC648,x
    lda $7EC688,x
    and #$DFFF
    sta $7EC688,x

    dex
    dex
    bpl -

.dont
    ; run hi-jacked instruction
    lda #$9DBF
    rts

warnpc !bank_80_free_space_end

org $82AC5A
    jsr (hook_equipment_screen_main_category_jump_table, X)

org $82ACEF
    jmp hook_tanks_dpad_response

org $82B03F
    jsr hook_beams_dpad_response
    nop
    nop

org $82B136
    jsr hook_suits_dpad_response
    ; beq to $B14E

org $82B4F5
    jsr hook_suits_move_higher
    nop
    nop

org $82B26A
    jmp hook_equipment_screen_selector

org $82B56B
    jsl hook_equipment_button_response_safetynet
    nop

org $8291B4
    jsl hook_load_equipment_menu

; Pause Menu Spritemap 10h, currently unused
; Origin size is 0005h
org $82C262
hook_pause_spritemap_smallbox:
    dw $0004
    
    dw $0005
    db $05
    dw $EEAE
    
    dw $01FB
    db $05
    dw $AEAE
    
    dw $0005
    db $FB
    dw $6EAE
    
    dw $01FB
    db $FB
    dw $2EAE
    
    ; 0 out the vanilla data for the un-needed 5th tile
    dw $0000
    db $00
    dw $0000

org $828215
    jsr hook_ppu_gameplay_setup
    
org $82936D
    jsr hook_unpause_loading

org $82A32D
    jsr hook_ppu_gameplay_setup

org $8291CA
    jsr hook_load_equip_menu

org $8291ED
    jsr hook_load_map

org !bank_82_free_space_start

config_classic:
    dw $0000

hook_unpause_loading:
    ;jsl $809A79 ; Hard re-initialize the HUD after we've messed with it.
    ldx #$0012
-
    ; Reinitialize the energy area of the HUD tiles from the base.
    lda $8098CB,x
    sta $7EC608,x
    lda $80990B,x
    sta $7EC648,x
    lda $80994B,x
    sta $7EC688,x
    dex
    dex
    bpl -
    stz $0A06   ; set previous health to invalid value, to trigger it to be redrawn
    dec $0A06
    jmp $A2E3 ;Hijacked


hook_ppu_gameplay_setup:
    ; lda #$04 from vanilla
    sta $210C ;hijacked (both 8215 and A32D are this instruction)
    sta $9C
    rts

hook_load_equip_menu:
    lda #$0014
hook_load_done:
    sta $9C
    lda #$0001 ;Hijacked
    rts

hook_load_map:
    lda #$0004
    bra hook_load_done


hook_equipment_screen_main_category_jump_table:
    ; 0 - reserve tank, 1 - beams, 2 - suits/misc, 3 - boots - all go to their vanilla entry points
    ; 4 - e-tanks is ours
    dw $AC70, $AFBE, $B0C2, $B150, hook_equipment_screen_category_etanks

hook_tanks_dpad_response:
    lda $0755
    and #$FF00
    bne +
    jsl dpad_enter_hud
    jmp $AD08
+
    jmp $ACF7

hook_beams_dpad_response:
    jsl dpad_enter_hud
    bcs +
    lda $12
    sta $0755
+
    rts

hook_suits_dpad_response:
    ; Just loaded A from $0755
    and #$ff00 ;Hijacked instruction
    beq .varia
    rts
.varia
    jsl dpad_enter_hud
    lda #$0000
    rts

hook_suits_move_higher:
    jsl dpad_enter_hud
    bcs +
    lda $12
    sta $0755
+
    rts

hook_equipment_screen_selector:
    lda $0755
    and #$00ff
    cmp #$0004
    bcc selector_go_back

    lda $0755
    xba
    and #$00ff
    inc
    tax
    
    and #$0008
    beq .no_adj
    ldy #$0007
    txa
    sec
    sbc #$0007
    bra .no_adj2

.no_adj
    ldy #$000f
    txa
.no_adj2
    asl #$3
    tax

    lda #$3600    ; Sprite priority 3, palette 3
    sta $03
    
    ;lda #$0014
    lda #$0010
    
    jsl $81891F   ; Draw sprite from pause menu spritemap

    jmp $B2A0

selector_go_back:
    lda $09A8   ; Hijacked instruction
    jmp $B26D

etanks_dpad_right:
    lda !etank_hud_tile_offset ; xxxE -> furthest right tank tile
    and #$000F
    cmp #$000E
    beq .no
    
    lda !current_etank_index
    inc a
    cmp !count_all_etanks     ; Don't go right past the total number of E-Tanks
    bcs .no

    sep #$20
    inc $0756
    rep #$20
    
    sec
    rts
.no
    clc
    rts

etanks_dpad_left:
    lda !etank_hud_tile_offset ;xxx2 -> furthest left tank tile
    and #$000F
    cmp #$0002
    beq .no
    
    lda config_classic
    bne .classic
    
    lda !current_etank_index
    dec a
    cmp !count_full_etanks     ; Don't go into full e-tanks
    bcc .no

.classic

    sep #$20
    dec $0756
    rep #$20
    
    sec
    rts
.no
    clc
    rts

etanks_dpad_down:
    lda !etank_hud_tile_offset    ; xx4x -> bottom row
    and #$00F0
    bne .no
    
    lda config_classic
    bne .classic
    
    ; Don't move down into full e-tanks
    lda $09C2
    cmp #$02BC     ; [Current energy] >= 700 means all 7 bottom row tanks are full
    bcs .no
.classic

    lda !current_etank_index
    sec
    sbc #$0007
    tax
    lda config_classic
    bne +
    cpx !count_full_etanks
    bcs +
    ldx !count_full_etanks
+
    sep #$10
    stx $0756
    rep #$10

.done
    sec
    rts

.no
; Don't go back to hook_equipment_screen_category_etanks
    lda #$0037
    jsl $809049
    pla ; Remove the JSR - this leaves a "php" on the stack, which $ABAD needs
    jmp $ABAD



etanks_dpad_up:
    lda !etank_hud_tile_offset ;xx0x -> top row
    and #$00F0
    beq .no

    lda !current_etank_index
    clc
    adc #$0008
-
    dec a
    cmp !count_all_etanks
    bcs -
    cmp !current_etank_index
    beq .no

    sep #$20
    sta $0756
    rep #$20
    
    sec
    rts
.no
    clc
    rts

; Controller input table - zero-terminated
hook_equipment_screen_category_etanks_controller_table:
    dw $0100
hook_equipment_screen_category_etanks_controller_table_func:
    dw etanks_dpad_right
    dw $0200, etanks_dpad_left
    dw $0400, etanks_dpad_down
    dw $0800, etanks_dpad_up
    dw $0080, etanks_a_button
    dw $0000

etanks_a_button:
    ; Is this a disabled E-Tank
    lda !current_etank_index
    cmp !count_enabled_etanks
    bcs .selected_disabled_tank

    jsl disable_tank

    bra .ret

.selected_disabled_tank
    jsl enable_tank

.ret
    clc
    rts

hook_equipment_screen_category_etanks:

    php
    rep #$30
    
    jsl etank_do_some_math

    ;for (short* x = hook_equipment_category_etanks_controller_table; *x; x += 2)
    ldx #$0000
    bra .start
-
    inx #$4
.start
    lda hook_equipment_screen_category_etanks_controller_table,x
    beq .ret

    ; All bits indicated must be newly pressed
    lda $8f
    beq .ret
    and hook_equipment_screen_category_etanks_controller_table,x
    cmp hook_equipment_screen_category_etanks_controller_table,x
    bne -

    ; Found a candidate
    jsr (hook_equipment_screen_category_etanks_controller_table_func, x)
    bcc .ret
    
    lda #$0037
    jsl $809049

.ret
    plp
    rts

warnpc !bank_82_free_space_end

org !bank_83_free_space_start

etank_do_some_math:
    php
    rep #$30
    
    lda $0755
    xba
    and #$00ff
    sta !current_etank_index    ; current_etank_index from equipment current item

    lda $09C2
    sta $4204
    sep #$20
    lda #$64
    sta $4206
    pha
    pla
    pha
    pla
    rep #$20
    lda $4214   
    sta !count_full_etanks      ; count_full_etanks = current_health / 100

    lda $09C4
    sta $4204
    sep #$20
    lda #$64
    sta $4206
    pha
    pla
    pha
    pla
    rep #$20
    lda $4214
    sta !count_enabled_etanks   ; count_enabled_etanks = maximum_health / 100
    clc
    adc !num_disabled_etanks
    sta !count_all_etanks       ; count_all_tanks = count_enabled_tanks + num_disabled_tanks

    lda !current_etank_index
    asl a
    tax
    lda $809CCE,X ; There's a nice table here of E-Tank tile offsets that are convenient for our purposes
    sta !etank_hud_tile_offset
    
    plp
    rtl

; Carry set = successful, carry clear = not successful
dpad_enter_hud:
    ; Need to save this in case it's being used.
    lda $12
    pha
    jsl etank_do_some_math
    pla
    sta $12
    
    lda.l config_classic
    bne .classic
    
    ; Can we move up here at all.
    lda !count_all_etanks
    cmp !count_full_etanks
    beq .no
    bra .which

.classic
    ; Have we any tanks at all
    lda !count_all_etanks
    beq .no

.which
    ; Are any not-full etanks enabled
    lda !count_full_etanks
    cmp !count_enabled_etanks
    bcs .all_disabled
    
    ; Select leftmost enabled tank (A button disables all tanks)
    lda !count_full_etanks
    cmp !count_all_etanks
    xba
    and #$ff00
    ora #$0004
    bra .yes
    
.all_disabled
    ; Select rightmost disabled tank (A button enables all tanks)
    lda !count_all_etanks
    dec a
    xba
    and #$ff00
    ora #$0004

.yes
    sta $0755
    lda #$0037
    jsl $809049 ; SFX "cursor moved"
    stz $0A06  ; Force the HUD to redraw energy - this will cause the HUD to update with de-prioritized tiles so the cursor can draw over it
    dec $0A06
    
    sec
    rtl

.no
    clc
    rtl

disable_tank:
    ; Sanity check: can we actually disable this tank?
    lda.l config_classic
    bne .classic
    lda $09C4
    sec
    sbc #$0064
    cmp $09C2
    bcc tank_swap_done ; (max health - 100) < current health = bad times
.classic

    ; Disable 1 e-tank
    lda $09C4
    sec
    sbc #$0064
    sta $09C4
    
    inc !num_disabled_etanks
    dec !count_enabled_etanks
    
    lda !current_etank_index
    cmp !count_enabled_etanks
    bcc disable_tank ; Repeat until we've disabled all tanks above and including the selected tank
    
    lda.l config_classic
    beq .skipclamp
    
    lda $09C4
    cmp $09C2
    bcs .skipclamp  ; max health < current health = need to clamp

    sta $09C2
.skipclamp

    bra tank_swap_good

enable_tank:
    lda !num_disabled_etanks  ; is number of disabled ETanks non-zero?
    beq tank_swap_done

    ; Decrement disabled ETank count
    dec !num_disabled_etanks
    inc !count_enabled_etanks

    ; Increase max health by 100
    lda $09C4
    clc
    adc #$0064
    sta $09C4
    
    lda !current_etank_index
    cmp !count_enabled_etanks
    bcs enable_tank ; Repeat until we've enabled all tanks below and including the current tank

tank_swap_good:
    lda #$0038
    jsl $809049 ; SFX "menu selection"

tank_swap_done:
    stz $0A06   ; set previous health to invalid value, to trigger it to be redrawn
    dec $0A06
    rtl

hook_equipment_button_response_safetynet:
    ; If we moved into the e-tank area, don't allow beams/suits/misc/boots to process an A button press and do possible weird things
    lda $0755
    and #$00ff
    cmp #$0004
    bcc .okay
    lda #$0000
    rtl
.okay
    ; Hijacked code
    lda $8F
    bit #$0080
    rtl


hook_load_equipment_menu:
    jsl $82AC22    ;Hijacked code
    
    ; Write the modified tile to VRAM at $2AE0
    ldx $0330
    lda #$0020
    sta $d0,x
    inx
    inx
    lda.w #tile_modified_map_cursor
    sta $d0,x
    inx
    inx
    lda #$00b6
    sta $d0,x
    inx
    lda #$2AE0
    sta $d0,x
    inx
    inx
    stx $0330
    rtl

warnpc !bank_83_free_space_end

org $B6FE60
tile_modified_map_cursor:
    db $00, $00, $00, $00, $3F, $00, $20, $00, $2F, $00, $28, $00, $28, $00, $28, $00
    db $00, $00, $00, $00, $3F, $00, $20, $00, $2F, $00, $28, $00, $28, $00, $28, $00

