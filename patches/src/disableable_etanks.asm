!bank_80_free_space_start = $80E3C0
!bank_80_free_space_end = $80E440

!bank_82_free_space_start = $82F830
!bank_82_free_space_end = $82F9F6

!bank_85_free_space_start = $85AD00
!bank_85_free_space_end = $85AE13

!current_etank_index = $12
!count_full_etanks = $14
!count_enabled_etanks = $16
!count_all_etanks = $18
!etank_hud_tile_offset = $1A


incsrc "constants.asm"

org $809698
    jsr hook_hud_begin
    nop
    nop

org $809BEE
    jsr hook_draw_tanks

org $809BFB
    jsr hook_blink_selected_etank

org !bank_80_free_space_start
hook_hud_begin:
    php
    rep #$20
    lda $0998	; If [game state] = Fh (pause menu)
    and #$00ff
    cmp #$000f
    bne .no

    plp		; Enable BG3 + Sprites in the HUD
    lda #$14
    sta $212C
    rts
.no
    ;Hijacked code
    plp
    lda #$04
    sta $212C
    rts

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
    ; run hi-jacked instruction
    lda #$9DBF
    rts

hook_blink_selected_etank:
    ; If we are in the pause menu:
    lda $0998
    and #$00ff
    cmp #$000F
    bne .done
    
    ; Equipment screen selected
    lda $0727
    cmp #$0001
    bne .done
    
    ; Make the entire energy area of the HUD not be priority so that we can draw sprites on it
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
    
.done
    lda #$9DD3   ; Hijacked
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


org !bank_82_free_space_start

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
    asl a
    asl a
    tax
    
    lda positions_etank_selector_sprite_x, X
    sta $12
    lda positions_etank_selector_sprite_y, x
    sta $14
    
    lda $12
    tax
    lda $14
    tay
    
    lda #$3600	; Sprite priority 3, palette 3
    sta $03
    
    ;lda #$0014
    lda #$0010
    
    jsl $81891F	; Draw sprite from pause menu spritemap

    jmp $B2A0

selector_go_back:
    lda $09A8   ; Hijacked instruction
    jmp $B26D

positions_etank_selector_sprite_x:
    ; X, Y position
    dw $0008
positions_etank_selector_sprite_y:
    dw $000F
    dw $0010, $000F
    dw $0018, $000F
    dw $0020, $000F
    dw $0028, $000F
    dw $0030, $000F
    dw $0038, $000F
    dw $0008, $0007
    dw $0010, $0007
    dw $0018, $0007
    dw $0020, $0007
    dw $0028, $0007
    dw $0030, $0007
    dw $0038, $0007

etanks_dpad_right:
    lda !etank_hud_tile_offset ; xxxE -> furthest right tank tile
    and #$000F
    cmp #$000E
    beq .no
    
    lda !current_etank_index
    inc a
    cmp !count_all_etanks     ; Don't go right past the total number of E-Tanks
    bcs .no

    lda $0755
    clc
    adc #$0100
    sta $0755
    
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
    
    lda !current_etank_index
    dec a
    cmp !count_full_etanks     ; Don't go into full e-tanks
    bcc .no

    sep #$20
    dec $0756
    rep #$20
    
    sec
    rts
.no
    clc
    rts

etanks_dpad_down:
    lda !etank_hud_tile_offset	; xx4x -> bottom row
    and #$00F0
    bne .no
    
    ; Don't move down into full e-tanks
    lda $09C2
    cmp #$02BC	; [Current energy] >= 700 means all 7 bottom row tanks are full
    bcs .no

    lda $0755
    xba
    and #$00ff
    sec
    sbc #$0007
    cmp !count_full_etanks
    bcs +
    lda !count_full_etanks
+
    sta $0756
    
    sec
    rts
.no
    clc
    rts

etanks_dpad_up:
    lda !etank_hud_tile_offset ;xx0x -> top row
    and #$00F0
    beq .no

    lda !current_etank_index
    clc
    adc #$0007
    cmp !count_all_etanks
    bcc +
    lda !count_all_etanks
    dec a
    cmp !current_etank_index
    beq .no
+
    sep #$20
    sta $0756
    rep #$20
    
    sec
    rts
.no
    clc
    rts

hook_equipment_screen_category_etanks:

    php
    rep #$30
    
    jsl etank_do_some_math

    lda $8f
    bit #$0100	; P1 D-Pad Right
    beq .hook_etank_not_right
    
    jsr etanks_dpad_right

    bra .hook_etank_done

.hook_etank_not_right

    lda $8f
    bit #$0200	; P1 D-Pad Left
    beq .hook_etank_not_left
    
    jsr etanks_dpad_left

    bra .hook_etank_done

.hook_etank_not_left

    lda $8F
    bit #$0400	; P1 D-Pad Down
    beq .hook_etank_not_down
    
    jsr etanks_dpad_down
    bcs .hook_etank_done

.hook_etank_down_from_hud
    lda #$0037
    jsl $809049
    jmp $ABAD

.hook_etank_down_from_hud_done
    clc
    bra .hook_etank_done

.hook_etank_not_down

    lda $8F
    bit #$0800	; P1 D-Pad Up
    beq .hook_etank_not_up
    
    jsr etanks_dpad_up

    bra .hook_etank_done

.hook_etank_not_up

    lda $8f
    bit #$0080	; P1 Button A
    beq .ret


    ; Is this a disabled E-Tank
    lda !current_etank_index
    cmp !count_enabled_etanks
    bcs .selected_disabled_tank

    jsl disable_tank

    bra .ret

.selected_disabled_tank
    jsl enable_tank
    bra .ret

.hook_etank_done
    bcc .ret
    
    lda #$0037
    jsl $809049

.ret
    plp
    rts

warnpc !bank_82_free_space_end

org !bank_85_free_space_start

etank_do_some_math:
    php
    rep #$30
    
    lda $0755
    xba
    and #$00ff
    sta !current_etank_index	; current_etank_index from equipment current item

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
    sta !count_full_etanks	; count_full_etanks = current_health / 100

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
    sta !count_enabled_etanks	; count_enabled_etanks = maximum_health / 100
    clc
    adc !num_disabled_etanks
    sta !count_all_etanks	; count_all_tanks = count_enabled_tanks + num_disabled_tanks

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
    
    ; Can we move up here at all.
    lda !count_all_etanks
    cmp !count_full_etanks
    beq .no
    
    ; Are any not-full etanks enabled
    lda !count_full_etanks
    cmp !count_enabled_etanks
    bcs .all_disabled
    
    ; Select leftmost enabled tank (A button disables all tanks)
    lda !count_full_etanks
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
    lda $09C4
    sec
    sbc #$0064
    cmp $09C2
    bcc tank_swap_done ; (max health - 100) < current health = bad times

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

warnpc !bank_85_free_space_end

org $B6FE60
tile_modified_map_cursor:
	db $00, $00, $00, $00, $3F, $00, $20, $00, $2F, $00, $28, $00, $28, $00, $28, $00
	db $00, $00, $00, $00, $3F, $00, $20, $00, $2F, $00, $28, $00, $28, $00, $28, $00

