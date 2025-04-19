arch snes.cpu
lorom

!bank_81_freespace_start = $81F100  ; TODO: remove this (not being used at the moment)
!bank_81_freespace_end = $81F140
!bank_82_freespace_start = $82F70F
!bank_82_freespace_end = $82F810
!bank_85_freespace_start = $85A280  ; must match reference in item_dots_disappear.asm and fix_kraid_hud.asm
!bank_85_freespace_end = $85A880
!etank_color = $82FFFE   ; must match addess customize.rs (be careful moving this, will probably break customization on old versions)
!bank_a7_freespace_start = $A7FFC0
!bank_a7_freespace_end = $A7FFE0

incsrc "constants.asm"

!tiles_2bpp_address = $B200
!tiles_2bpp_bank = $009A

!backup_area = $1F62
!map_switch_direction = $1F66
!unexplored_gray = #$294a
;!unexplored_gray = #$18c6
;!unexplored_light_gray = #$35ad
!unexplored_light_gray = #$4631
!area_explored_mask = $702600


; pause map colors for palettes 2 (explored) and 6 (unexplored):
; 
; 0: transparent (black, but grid lines from layer 2 can show through)
; 1: cool room color (dark gray for unexplored, or dark area-themed color for explored)
; 2: hot room color (ligh gray for unexplored, or light area-themed color for explored)
; 3: white (walls/passages)
; 4: black
; 5: unused
; 6: orange door, tourian arrow
; 7: pink door
; 8: blue: maridia arrow
; 9: yellow: wrecked ship arrow
; 10: red: norfair arrow
; 11: purple: crateria arrow
; 12: white (item dots)
; 13: black (door lock shadows covering wall)
; 14: green door, brinstar arrows
; 15: gray door
;
; Palette 3 is used for partially revealed tiles (i.e. showing outline of visited rooms),
; and essentially replaces all colors with black except for 3 (which remains white) and
; 13, which becomes white in order to not give away the presence of a door lock.


;;; Hijack map usages of area ($079F) with new area ($1F5B)
org $8085A7  ; Load mirror of current area's map explored
    ldx $1F5B

org $80858F  ; Load mirror of current area's map explored
    lda $1F5B

org $8085C9  ; Mirror current area's map explored
    lda $1F5B

org $8085E6  ; Mirror current area's map explored
    ldx $1F5B

org $82941B  ; Updates the area and map in the map screen
    lda $1F5B

org $829440  ; Updates the area and map in the map screen
    lda $1F5B

org $829475  ; Updates the area and map in the map screen
    ldx $1F5B

org $82952D  ; Draw room select map
    lda $1F5B

org $829562  ; Draw room select map
    ldx $1F5B

org $82962B  ; Draw room select map area label
    lda $1F5B

org $829ED5  ; Determining map scroll limits
    lda $1F5B

org $829F01  ; Determining map scroll limits
    lda $1F5B

org $90A9BE  ; Update mini-map
    lda $1F5B

org $90AA73  ; Update HUD mini-map tilemap
    lda $1F5B

org $90AA78  ; Update HUD mini-map tilemap
    adc $1F5B

org $848C91  ; Activate map station
    ldx $1F5B

org $8FC90C  ; Tourian first room gives area map (TODO: change this)
    ldx $1F5B

org $84B19C  ; At map station, check if current area map already collected
    ldx $1F5B

org $82E488
    bra +    ; skip reloading BG3 tiles in door transition
org $82E492
+

;;; Hijack code that loads room state, in order to populate map area
org $82DEF7
    jsr load_area_wrapper

;org $828D08
;org $828D4B
org $828D44
    jsr pause_start_hook_wrapper

; org $82936A
org $80A15F
    jsl pause_end_hook

; hook library background load in Kraid Room:
org $82E63E
    jsl kraid_room_load_hook
    nop

org $829130 : jsr draw_samus_indicator
org $82915A : jsr draw_samus_indicator
org $829200 : jsr draw_samus_indicator
org $82935B : jmp draw_samus_indicator
org $82910A : jsr (PauseRoutineIndex,x)

org $829125
    jsr check_start_select

; Use consistent version of map scrolling setup so we don't have to patch both versions of it:
org $829E27
    jsl $829028
    rts

org $82903B
    jsr horizontal_scroll_hook

org $829E38  ; TODO: remove this (should be unused?)
    jsr horizontal_scroll_hook

org $82E7C9
    jsr load_tileset_palette_hook_wrapper
    nop : nop : nop : nop

org $82E1F7
    jsr palette_clear_hook

;; Don't preserve palette 7 color 1 (used for FX), let it fade to black:
;org $82E21E
;    nop : nop : nop  ; was: STA $C23A

;org $82E464
;org $82E55F
;org $82E780
org $82E764
    jsr door_transition_hook_wrapper

org $82E4A2
    jsr load_target_palette_hook_wrapper

org $90AB4A
    jsl samus_minimap_flash_hook : nop : nop

; Indicate Samus position on HUD by flashing tile palette 0 instead of palette 7
org $90AB56
    AND #$E3FF     ; was: ORA #$1C00

; Use palette 3 for full ETanks (instead of palette 2)
org $809BDC
    LDX #$2C31     ; was: LDX #$2831


; Use palette 7 (gray/white, same as unexplored tiles) for fixed HUD (e.g. "ENERGY"),
; For blank tiles, use palette 3, which has an opaque black (color 3).
org $80988B
dw $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $3C1D, $3C1D, $3C1D, $3C1D, $3C1D, $3C1C,
   $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $3C12, $3C12, $3C23, $3C12, $3C12, $3C1E,
   $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $2822, $2822, $2823, $2813, $3C14, $3C1E,
   $0C0F, $3C0B, $3C0C, $3C0D, $3C32, $0C0F, $3C09, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $0C0F, $3C12, $3C12, $A824, $2815, $3C16, $3C1E

; Use palette 7 (gray/white, same as unexplored tiles) for HUD digits
org $809DBF : dw $3C00, $3C01, $3C02, $3C03, $3C04, $3C05, $3C06, $3C07, $3C08, $3C09
org $809DD3 : dw $3C00, $3C01, $3C02, $3C03, $3C04, $3C05, $3C06, $3C07, $3C08, $3C09

;; Use palette 7 (instead of 6) when initializing (clearing) FX tilemap:
;org $82E569 : lda #$1C0F   ; was: lda #$184E
;org $80A214 : lda #$1C0F   ; was: lda #$184E
;;org $80A2F7 : dw $1C0F   ; was: dw $184E  (not doing this, since it's overwritten in hud_expansion_opaque.asm instead)

; For message boxes, skip modifying palette 6:
org $858150 : rep $19 : nop

; Use palette 0 for full auto reserve
org $80998B             
    dw  $2033, $2046,
        $2047, $2048,
        $A033, $A046

; Use palette 7 for empty auto reserve
org $809997             
dw $3C33, $3C46,
   $3C47, $3C48,
   $BC33, $BC46

;; Use palette 7 for palette blends (FX: water, lava, etc.)
;org $89AB62 : STA $7EC03A   ; was: STA $7EC032
;org $89AB6A : STA $7EC03C   ; was: STA $7EC034
;org $89AB72 : nop : nop : nop : nop   ; was: STA $7EC036
;;org $89AB72 : STA $7EC03E   ; was: STA $7EC036
;org $89AB7B : nop : nop : nop : nop   ; was: STA $7EC036
;
;org $89AC05 : STA $7EC23A   ; was: STA $7EC232
;org $89AC0D : STA $7EC23C   ; was: STA $7EC234
;org $89AC15 : nop : nop : nop : nop   ; was: STA $7EC236
;;org $89AC15 : STA $7EC23E   ; was: STA $7EC236
;org $89AC1E : nop : nop : nop : nop   ; was: STA $7EC236

org $8291F9
    jsr load_map_screen_wrapper

org $8291D0
    jsr load_equipment_screen_wrapper

;;; Put new code in free space at end of bank $82:
org !bank_82_freespace_start

; This function must come first, in order to be at the address expected in pause_menu_objectives.asm
switch_map_area_wrapper:
    jsl switch_map_area
    rts

; when switching from equipment screen to map screen, restore certain palette colors
load_map_screen_wrapper:
    jsl load_map_screen
    rts

load_equipment_screen_wrapper:
    jsl load_equipment_screen
    rts

load_area_wrapper:
    jsl load_area
    rts

vram_transfer_wrapper:
    jsr $E5EB
    rtl

PauseRoutineIndex:
	DW $9120, $9142, $9156, $91AB, $9231, $9186, $91D7, $9200	;same as $9110
	DW $9156, switch_map_area_wrapper, $9200		;fade out / map construction / fade in

pause_start_hook_wrapper:
    jsl pause_start_hook
    rts

pause_end_hook:
    jsl $82E97C  ; run hi-jacked instruction (load library background)
    lda !backup_area
    sta $1F5B  ; restore map area
    jsl $80858C ; restore map explored bits
    jsl load_bg3_map_tiles_wrapper
    jsl load_bg3_map_tilemap_wrapper
    jsl set_hud_map_colors_wrapper
    rtl

check_start_select:
    php
    rep #$30

    stz !map_switch_direction
    lda $8F        ; load newly pressed input
    bit #$6000
    bne .switch      ; if select/Y (next map) is not newly pressed, continue as normal

    bit #$0040
    beq .skip      ; if X (previous map) is not newly pressed, continue as normal
    lda #$0001
    sta !map_switch_direction

.switch:
    ; switch to next area map:
    lda #$0037
    jsl $809049    ; play sound "move cursor"

    LDA #$0000            ;\
    STA $0723             ;} Screen fade delay = 0
    LDA #$0001
    STA $0725 
    lda #$0008      ; fade out
    sta $0727

.skip:
    plp
    jsr $A5B7      ; run hi-jacked code (handle pause screen start button)
    rts

area_palettes_explored:
    dw $6c12  ; Crateria
    dw $0240  ; Brinstar
    dw $0017  ; Norfair
    dw $0230  ; Wrecked Ship
    dw $7583  ; Maridia
    dw $0195  ; Tourian

area_palettes_explored_light:
    dw $7dfb  ; Crateria
    dw $332c  ; Brinstar
    dw $319f  ; Norfair
    dw $2ef7  ; Wrecked Ship
    dw $7e8c  ; Maridia
    dw $323d  ; Tourian

draw_samus_indicator:
	lda !backup_area
    cmp $1F5B 
    bne .skip		; check if area shown is the same area as samus
	jsr $B9C8       ; if so, draw the indicator showing where samus is.
.skip:
    rts

horizontal_scroll_hook:
    ; round BG1 scroll X to a multiple of 8, to make grid lines consistently align with tiles:
    sbc #$0080   ; run hi-jacked instruction
    and #$FFF8
    rts

load_tileset_palette_hook_wrapper:
    jsl load_tileset_palette_hook
    rts

palette_clear_hook:
    lda $C016  ; preserve explored white color (2bpp palette 2, color 3)
    sta $C216

    lda $C03A  ; preserve unexplored gray color (2bpp palette 6, color 1)
    sta $C23A

    lda $C03C  ; preserve unexplored light gray color (2bpp palette 6, color 2)
    sta $C23C

    lda $C03E  ; preserve unexplored white color (2bpp palette 6, color 2)
    sta $C23E

    ; Preserve full Auto reserve color, PB door, Samus HUD indicator, etc.: palette 0, color 1-3
    lda $C002
    sta $C202
    lda $C004
    sta $C204
    lda $C006
    sta $C206

    lda $C014  ; run hi-jacked instruction
    rts

load_target_palette_hook_wrapper:
    jsl load_target_palette_hook
    rts

door_transition_hook_wrapper:
    jsl door_transition_hook
    rts

reset_pause_animation_wrapper:
    jsr $A0F7    ; Reset pause menu animations
    rtl

determine_map_scroll_wrapper:
    jsr $9EC4    ; Determine map scroll limits
    rtl

print pc
warnpc !bank_82_freespace_end

org !bank_85_freespace_start

; this function must go first, to match the reference in item_dots_disappear.asm
load_bg3_map_tilemap_wrapper:
    jsr load_bg3_map_tilemap
    rtl

warnpc $85A290
org $85A290
; must match the reference in fix_kraid_hud.asm
load_bg3_map_tiles_wrapper:
    jsr load_bg3_map_tiles
    rtl

clear_hud_minimap:
    ; clear HUD minimap during area transitions
    LDX #$0000             ;|
    lda #$3C50
.clear_minimap_loop:
    STA $7EC63C,x          ;|
    STA $7EC67C,x          ;} HUD tilemap (1Ah..1Eh, 1..3) = 3C50h
    STA $7EC6BC,x          ;|
    INX                    ;|
    INX                    ;|
    CPX #$000A             ;|
    BMI .clear_minimap_loop

    ; update VRAM for HUD
    LDX $0330       ;\
    LDA #$00C0      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    LDA #$C608      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;} Queue transfer of $7E:C608..C7 to VRAM $5820..7F (HUD tilemap)
    LDA #$007E      ;|
    STA $D0,x       ;|
    INX             ;|
    LDA #$5820      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    STX $0330       ;/
    
    rtl

;;; X = room header pointer
load_area:
    phy

    ;;; Load the new area number (for use in map) into $1F5B
    ldx $07bb      ; x <- room state pointer
    lda $8F0010,x
    tax            ; x <- extra room data pointer
    lda $B80000,x  ; a <- [extra room data pointer]
    and #$00FF
    sta $1F5B

    ; mark area as explored (determinines set of valid area maps to cycle through in pause menu):
    jsl $80818E    ; convert map area to bitmask
    lda $05E7      ; load bitmask
    ora !area_explored_mask    ; combine with area explored mask
    sta !area_explored_mask    ; update area explored mask

    ply
    ldx $07bb      ; run hi-jacked instruction: x <- room state pointer
    rtl

door_transition_hook:
    jsr set_hud_map_colors
    lda #$0008   ; run hi-jacked instruction
    rtl

pause_start_hook:
    lda $1F5B
    sta !backup_area  ; back up map area
    jsr set_hud_map_colors
    jsr update_pause_map_palette
    jsl load_bg3_map_tiles_wrapper
    ;jsr remove_samus_hud_indicator
    jsl $8085C6  ; save current map explored bits
    ;jsr $8D51  ; run hi-jacked instruction
    ;inc $0998  ; run hi-jacked instruction
    stz $05FF  ; run hi-jacked instruction
    rtl

load_tileset_palette_hook:
    ; run hi-jacked instruction:
    jsl $80B0FF
    dl $7EC200

    jsr set_hud_map_colors
    jsr load_target_palette
    rtl

load_map_screen:
    ; run hi-jacked instruction:
    sta $0763

    ; palette 2:
    lda $B6F05C
    sta $7EC05C
    lda $B6F05E
    sta $7EC05E

    ; palette 3:
    lda $B6F06E
    sta $7EC06E
    lda $B6F07A
    sta $7EC07A

    ; palette 6:
    lda $B6F0CC
    sta $7EC0CC
    lda $B6F0D6
    sta $7EC0D6
    lda $B6F0CE
    sta $7EC0CE
    lda $B6F0DE
    sta $7EC0DE

    ; palette 7:
    lda $B6F0E2
    sta $7EC0E2
    lda $B6F0E4
    sta $7EC0E4
    lda $B6F0E6
    sta $7EC0E6
    lda $B6F0EC
    sta $7EC0EC
    lda $B6F0F6
    sta $7EC0F6
    lda $B6F0FA
    sta $7EC0FA
    lda $B6F0FC
    sta $7EC0FC

    ; Partially revealed tiles: black color for item dots, door locks
    lda #$0000
    sta $7EC07A
    sta $7EC06E

    ; Load map tile graphics
    jsl $828E75
    rtl

load_equipment_screen:
    ; run hi-jacked instruction:
    sta $0725

    ; Fix color used for pink doors on map screen
    lda #$6E7A
    sta $7EC0CE
    ; Fix color used for green doors on map screen
    lda #$5EF7
    sta $7EC05C
    ; Fix color used for gray doors on map screen
    lda #$318C
    sta $7EC0DE
    sta $7EC05E

    ; Fix colors used for partially revealed tiles on map screen
    lda #$7FFF
    sta $7EC07A
    lda #$5EF7
    sta $7EC06E

    ; Palette 7 (unexplored map colors):
    lda #$7FFF
    sta $7EC0E2
    lda #$4A52
    sta $7EC0E4
    sta $7EC0F6
    lda #$318C
    sta $7EC0E6
    sta $7EC0EC
    lda #$7FFF
    sta $7EC0FA
    lda #$6318
    sta $7EC0FC

    ; transfer equipment screen tile graphics to VRAM
    LDA #$0080
    STA $2115  ; video port control
    LDA #$0000
    STA $2116  ; VRAM (destination) address = $0000

    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    
    lda #$8000 ; source address = $8000
    sta $4312

    ; Set source bank to $B6:
    lda #$00B6
    sta $4314

    lda #$6000
    sta $4315 ; transfer size = $6000 bytes

    sep #$30
    lda #$02
    sta $420B  ; perform DMA transfer on channel 1
    rtl

load_target_palette_hook:
    jsr load_target_palette
    lda #$E4A9   ; run hi-jacked instruction
    rtl

load_target_palette:
    ; Prevent HUD map colors from gradually changing (e.g. to blue/pink) during door transition:
    lda $7EC01A  ; etank color (pink): palette 3, color 1 
    sta $7EC21A

    lda $7EC012  ; explored color (area-themed): palette 2, color 1
    sta $7EC212

    lda $7EC014  ; explored light color (area-themed): palette 2, color 2
    sta $7EC214

    lda $7EC016  ; explored white: palette 2, color 3
    sta $7EC216

    lda !unexplored_gray
    sta $7EC23A

    lda !unexplored_light_gray
    sta $7EC23C

    lda #$7FFF
    sta $7EC23E

    rts

switch_map_area:
    lda !map_switch_direction
    beq .next
    jsr prev_area
    jmp .update
.next:
    jsr next_area
.update:
    jsr update_pause_map_palette
	jsl $80858C     ;load explored bits for area
	lda $7ED908,x : and #$00FF : sta $0789	;set flag of map station for next area (TODO: remove this, should be unnecessary now.)
    jsl $8293C3		;update area label and construct new area map

    lda $1F5B
    cmp !backup_area
    beq .orig_area
    jsr simple_scroll_setup  ; for map in different area, set scrolls without using samus position
    bra .done
.orig_area:
    jsl $829028     ;set map scroll boundaries and screen starting position like vanilla, using samus position
.done:

    LDA #$0000             ;\
    STA $0723             ;} Screen fade delay = 0

    inc $0727
    rtl


next_area:
    lda $1F5B
    inc
    cmp #$0006
    bne .done
    lda #$0000
.done:
    sta $1F5B

    jsl $80818E     ; convert map area to bitmask
    lda $05E7       ; load bitmask
    and !area_explored_mask     ; test if area is explored
    beq next_area   ; if not, skip this area and try the next one.

    rts

prev_area:
    lda $1F5B
    dec
    cmp #$ffff
    bne .done
    lda #$0005
.done:
    sta $1F5B

    jsl $80818E     ; convert map area to bitmask
    lda $05E7       ; load bitmask
    and !area_explored_mask     ; test if area is explored
    beq prev_area   ; if not, skip this area and try the next one.

    rts

update_pause_map_palette:
    lda $1F5B
    asl
    tax

    ; Set unexplored gray color: palette 7, color 1
    lda !unexplored_gray
    sta $7EC0E2

    ; Set unexplored light gray color: palette 7, color 2
    lda !unexplored_light_gray
    sta $7EC0E4

    ; Set unexplored white color: palette 7, color 3
    lda #$FFFF
    sta $7EC0E6

    ; Set explored color based on area: palette 2, color 1
    lda area_palettes_explored, x
    sta $7EC042

    ; Set light explored color based on area: palette 2, color 2
    lda area_palettes_explored_light, x
    sta $7EC044

    ; Set explored white: palette 2, color 3
    lda #$7FFF
    sta $7EC046

    rts

samus_minimap_flash_hook:
    lda $0998
    cmp #$000C
    bne .normal

    ; Paused: skip showing Samus indicator:
    lda #$0001
    rtl
    
    ; Run hi-jacked instructions (use frame counter to determine whether to show Samus indicator)
.normal    
    lda $05B5
    and #$0008 

    rtl

set_hud_map_colors_wrapper:
    jsr set_hud_map_colors
    rtl

set_hud_map_colors:
    ; Set colors for HUD map:
    lda $1F5B
    asl
    tax

    ; Set unexplored gray: palette 7, color 1
    lda !unexplored_gray
    sta $7EC03A

    ; Set unexplored light gray: palette 7, color 2
    lda !unexplored_light_gray
    sta $7EC03C

    ; Set unexplored white: palette 7, color 3
    lda #$7FFF
    sta $7EC03E

    ; Set explored color based on area: palette 2, color 1
    lda.l area_palettes_explored, x
    sta $7EC012

    ; Set explored light color based on area: palette 2, color 2
    lda.l area_palettes_explored_light, x
    sta $7EC014

    ; Set explored white: palette 2, color 3
    lda #$7FFF
    sta $7EC016

    ; Set palette 3, color 1 to pink color for full E-tank energy squares
    ; lda #$48FB
    lda !etank_color
    sta $7EC01A

    ; Set palette 3, color 2 to white color for full E-tank energy squares
    lda #$7FFF
    sta $7EC01C

    rts


simple_scroll_setup:
    ; Like $829028 but without using Samus position, just midpoints.
    jsl reset_pause_animation_wrapper
    jsl determine_map_scroll_wrapper
    LDA $05AE    ;\
    SEC          ;|
    SBC $05AC    ;|
    LSR A        ;|
    CLC          ;} BG1 X scroll = midpoint([map min X scroll], [map max X scroll]) - 80h
    ADC $05AC    ;|
    SEC          ;|
    SBC #$0080   ;|
    AND #$FFF8   ;|
    STA $B1      ;/

    LDA $05B2    ;\
    SEC          ;|
    SBC $05B0    ;|
    LSR A        ;|
    CLC          ;|
    ADC #$0010   ;|
    CLC          ;|
    ADC $05B0    ;|
    STA $12      ;} BG1 Y scroll = midpoint([map min Y scroll], [map max Y scroll]) - 60h rounded up to multiple of 8
    LDA #$0070   ;|
    SEC          ;|
    SBC $12      ;|
    AND #$FFF8   ;|
    EOR #$FFFF   ;|
    INC A        ;|
    STA $B3      ;/    

    RTS

hud_minimap_tile_addresses:
    dw $C63C, $C63E, $C640, $C642, $C644
    dw $C67C, $C67E, $C680, $C682, $C684
    dw $C6BC, $C6BE, $C6C0, $C6C2, $C6C4
    dw $0000

save_hud_tiles:
    ; copy the 15 currently visible HUD mini-map tiles to tiles $20-$2E (VRAM $4100),
    ; via temporary SRAM $702D00-$702DF0
    ldx #$0000
    ldy #$2D00
    
save_hud_gfx_loop:
    lda.l hud_minimap_tile_addresses,x
    beq .done
    phx
    tax
    lda $7E0000,x          ; A <- tilemap word from HUD mini-map
    and #$03FF             ; A <- within-room tile number
    sec
    sbc #$0050             ; A <- within-room tile number - $50
    asl
    clc
    adc !room_map_tile_gfx
    tax
    lda $E40000,x          ; A <- global tile number
    asl : asl : asl : asl
    clc
    adc #$8000
    tax  ; X = source address (in bank $E3) = $8000 + $10 * global tile number
    
    ; transfer 16 bytes (a single map tile) from bank $E3 to bank $70
    phb
    lda #$000F
    mvn $70,$E3
    plb
    plx

    inx : inx
    bra save_hud_gfx_loop
.done:

    LDX $0330       ;\
    LDA #$00F0      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    LDA #$2D00      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;} Queue transfer of $70:2D00..2DF0 to VRAM $4100
    LDA #$0070      ;|
    STA $D0,x       ;|
    INX             ;|
    LDA #$4100      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    STX $0330       ;/

    ldx #$0000
    lda #$0020
    sta $00         ; $00 <- destination tile number
save_hud_tilemap_loop:
    lda.l hud_minimap_tile_addresses,x
    beq .done
    phx
    tax

    lda $7E0000,x   ; A <- tilemap word from HUD mini-map
    and #$03FF
    cmp #$0050
    bcc .invariant  ; Skip modifying tiles with tile number <= $50

    lda $7E0000,x   ; A <- tilemap word from HUD mini-map
    and #$FC00
    ora $00         ; replace tile number with tile in $20-$2E
    sta $7E0000,x

.invariant:
    lda $00
    inc
    sta $00
    plx
    inx : inx
    bra save_hud_tilemap_loop
.done:

    LDX $0330    ;\
    LDA #$00C0   ;|
    STA $D0,x    ;|
    INX          ;|
    INX          ;|
    LDA #$C608   ;|
    STA $D0,x    ;|
    INX          ;|
    INX          ;} Queue transfer of $7E:C608..C7 to VRAM $5820..7F (HUD tilemap)
    LDA #$007E   ;|
    STA $D0,x    ;|
    INX          ;|
    LDA #$5820   ;|
    STA $D0,x    ;|
    INX          ;|
    INX          ;|
    STX $0330    ;/

    rts

load_bg3_tiles_door_transition:
    php

    ; Wait for NMI, to avoid VRAM writes sometimes getting lost,
    ; which could happen if NMI interrupts the code that queues up a write.
    jsl $808338
    jsr save_hud_tiles
    jsl $808338
    jsr load_bg3_map_tiles
    jsr load_bg3_map_tilemap

    ; run hi-jacked instructions:
    plp
    lda $0791
    and #$0003 
    rtl

load_bg3_map_tiles:
    phx
    phy
    php

    rep #$30
    ldx $07BB      ; x <- room state pointer
    lda $8F0010,x
    tax            ; x <- extra room data pointer
    lda $B80003,x
    sta $00        ; $00 <- room map tile graphics pointer (in bank $E4)
    sta !room_map_tile_gfx

    ; transfer room map tile graphics to $70:7400
    ldx $00
    ldy #$7400
gfx_transfer_loop:
    phx
    lda $E40000,x
    beq .done
    asl : asl : asl : asl
    clc
    adc #$8000
    tax

    ; transfer 16 bytes (a single map tile) from bank $E3 to bank $7E
    phb
    lda #$000F
    mvn $70,$E3
    plb

    plx
    inx : inx
    bra gfx_transfer_loop
.done:
    plx

    ; load room map tile graphics to VRAM:
    LDX $0330
    LDA #$0500
    STA $D0,x
    INX
    INX
    LDA #$7400
    STA $D0,x
    INX
    INX
    LDA #$0070
    STA $D0,x
    INX
    LDA $079B
    CMP #$A59F      ; Is this Kraid Room?
    BNE +
    LDA $0998
    CMP #$000D      ; Is the pause screen loading?
    BEQ +
    CMP #$000E      ; Is the pause screen loading?
    BEQ +
    CMP #$000F      ; Is it in pause screen?
    BEQ +
    LDA $099C
    CMP #$E2F7      ; Are we leaving Kraid Room?
    BEQ +
    ; For Kraid's Room gameplay (not paused), load tiles into $2280 instead.
    LDA #$2280
    BRA ++
+
    LDA #$4280
++
    STA $D0,x
    INX
    INX
    STX $0330

    plp
    ply
    plx
    rts

load_bg3_map_tilemap:
    php
    phx
    phy

    ldx $07BB      ; x <- room state pointer
    lda $8F0010,x
    tax            ; x <- extra room data pointer
    lda $B80005,x
    sta $02        ; $02 <- room map tilemap pointer (in bank $E4)

    ; load room map tilemap
    lda $07A3
    sec
    sbc #$0001
    sta $04        ; $04 <- current tilemap row index = room y coordinate - 1
    clc
    adc $07AB
    adc #$0002
    sta $06        ; $06 <- end tilemap row index = current tilemap row index + room height + 2

tilemap_transfer_row_loop:
    lda $07A1
    sec
    sbc #$0002
    sta $08        ; $08 <- current tilemap column index = room x coordinate - 1
    clc
    adc $07A9
    adc #$0004
    sta $0A        ; $0A <- end tilemap col index = current tilemap col index + room width + 4

tilemap_transfer_col_loop:
    lda $08
    cmp #$0020
    bcs .second_page

    lda $04
    asl : asl : asl : asl : asl
    clc
    adc $08
    asl      ; A <- (row index * 32 + col index) * 2
    bra .transfer_word

.second_page:
    lda $04
    asl : asl : asl : asl : asl
    clc
    adc $08
    adc #$03E0
    asl          ; A <- (row index * 32 + col index - 32) * 2 + 0x800
    bra .transfer_word

.transfer_word:
    tay    
    ldx $02
    lda $E40000,x
    inx : inx
    stx $02
    tyx
    sta $703040,x

    lda $08
    inc
    sta $08
    cmp $0A
    bne tilemap_transfer_col_loop

    lda $04
    inc
    sta $04
    cmp $06
    bne tilemap_transfer_row_loop

    ply
    plx
    plp
    rts

start_game_hook:
    jsl load_bg3_map_tiles_wrapper
    jsl load_bg3_map_tilemap_wrapper
    
    ; run hi-jacked instructions
    ldy #$0020
    ldx #$0000
    rtl

area_cross_hook:
    jsl $80858C  ; run hi-jacked instruction
    jsl clear_hud_minimap
    rtl

kraid_room_load_hook:
    ; wait for NMI, to ensure the vanilla BG3 VRAM transfer
    ; happens atomically with the update to the HUD mini-map
    phy
    jsl $808338
    ply

    jsl vram_transfer_wrapper

    pha
    phx
    ; load current HUD mini-map tiles to VRAM $2100:
    LDX $0330       ;\
    LDA #$00F0      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    LDA #$2D00      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;} Queue transfer of $70:2D00..2DF0 to VRAM $2100
    LDA #$0070      ;|
    STA $D0,x       ;|
    INX             ;|
    LDA #$2100      ;|
    STA $D0,x       ;|
    INX             ;|
    INX             ;|
    STX $0330       ;/

    jsr load_bg3_map_tiles    

    plx
    pla

    ; run hi-jacked instruction:
    sep #$20
    rtl

warnpc !bank_85_freespace_end

org $82DFC2
    jsl area_cross_hook

; Unexplored gray: palette 7, color 1
org $B6F03A : dw !unexplored_gray  ; 2bpp palette
org $B6F0E2 : dw !unexplored_gray  ; 2bpp palette

; Unexplored light gray: 2bpp palette 7, color 2
org $B6F03C : dw !unexplored_light_gray  ; 2bpp palette
org $B6F0E4 : dw !unexplored_light_gray  ; 2bpp palette

; Unexplored white: 2bpp palette 7, color 3
org $B6F03E : dw $7FFF  ; 2bpp palette
org $B6F0E6 : dw $7FFF  ; 2bpp palette

; Patch tile data for button letters. Changing the palettes to 3:
; This takes into account the changes to the tileset from hud_expansion_opaque.asm.
org $858426            
    dw $2CC0, ; A
       $2CC1, ; B
       $2CD7, ; X
       $2CD8, ; Y
       $2CFA, ; Select
       $2CCB, ; L
       $2CD1, ; R
       $2C0F  ; Blank

; Phantoon power-on palette:
;org $A7CA77 : dw #$48FB            ; 2bpp palette 2, color 3: pink color for E-tanks (instead of black)
;org $A7CA7B : dw !unexplored_gray   ; 2bpp palette 3, color 1: gray color for HUD dotted grid lines

org $A7CA7B : dw #$48FB            ; 2bpp palette 3, color 1: pink color for E-tanks
;org $A7CA97 : dw #$7FFF            ; 2bpp palette 6, color 3: white color for HUD text/digits

; hook start of game to load correct BG3 tiles based on room:
org $82806E
    jsl start_game_hook
    nop : nop

; Patch door transition code to reload BG3 tiles based on room:
org $82E46A : beq $1c
org $82E472 : beq $14
org $82E492
    jsl load_bg3_tiles_door_transition
    nop : nop

; Patch pause menu start to load BG1/2 tiles, including the expanded set of map tiles:
org $828E75
    ; Load 4bpp tiles for the area into VRAM:
    php
    rep #$30

    LDA #$0080
    STA $2115  ; video port control
    LDA #$0000
    STA $2116  ; VRAM (destination) address = $0000

    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    
    lda #$8000 ; source address = $8000
    sta $4312

    ; Set source bank to $E2:
    lda #$00E2
    sta $4314

    lda #$6000
    sta $4315 ; transfer size = $6000 bytes

    sep #$30
    lda #$02
    sta $420B  ; perform DMA transfer on channel 1

    lda $0998
    cmp #$0D
    bne .skip_load_bg3  ; only load BG3 tiles when first entering pause menu, not when switching screens

    ; load reduced set of BG3 tiles, to avoid overwriting FX tiles:
    ; VRAM $4000..4600 = [$9A:B200..] (standard BG3 tiles)
    LDA #$00
    STA $2116
    LDA #$40
    STA $2117
    LDA #$80
    STA $2115
    JSL $8091A9
    db $01, $01, $18, $00, $B2, $9A, $00, $0C
    LDA #$02
    STA $420B

    jsl load_bg3_map_tiles_wrapper

.skip_load_bg3
    plp
    rtl
warnpc $828EDA

; Use palette 4 instead of palette 2 or non-map pause menu content
; (to free up more colors in palette 2 for use in map tiles).
; More of this palette switching happens in map_tiles.rs.
org $82A63A : ORA #$1000   ; equipment screen: top of MAP
org $82A658 : ORA #$1000   ; equipment screen: bottom of MAP
org $82A676 : ORA #$1000   ; equipment screen: top of EXIT
org $82A694 : ORA #$1000   ; equipment screen: bottom of EXIT
org $82A7A8 : ORA #$1000   ; map screen: top of SAMUS
org $82A7C6 : ORA #$1000   ; map screen: bottom of SAMUS
org $82A7E4 : ORA #$1000   ; map screen: top of EXIT
org $82A802 : ORA #$1000   ; map screen: bottom of EXIT

org !etank_color : dw $48FB  ; default pink E-tank color

;; Skip map select after game over:
;; (Map select on that screen uses different code and wouldn't work correctly with our modifications.)
;org $81911A
;    jmp game_over_load
;
;org !bank_81_freespace_start
;game_over_load:
;    lda $0952
;    jsl $818085  ; load from save slot
;    jsl $80858C
;    LDA #$0006
;    STA $0998    ; game state = 6 (loading game)
;    RTS
;warnpc !bank_81_freespace_end

; Use palette 7 instead of 3 when mini-map is disabled (during boss fights)
org $90A7F1
    ORA #$3C00   ; was: ORA #$2C00

; Make slope tiles $10 and $11 have the functionality that tile $28 does in vanilla,
; to trigger automatically exploring tile above Samus.
; (Tile $11 is used in Crocomire Speedway, the heated version of $10 used in Terminator Room.)
org $90AAFD
    AND #$83FE   ; was: AND #$01FF
    CMP #$0010

;; Kraid load BG3 from area-specific tiles:
;org $A7C78B : lda #!tiles_2bpp_address
;org $A7C790 : jsr get_area_bg3_bank
;org $A7C7B1 : lda #!tiles_2bpp_address+$400
;org $A7C7B6 : jsr get_area_bg3_bank
;org $A7C7D7 : lda #!tiles_2bpp_address+$800
;org $A7C7DC : jsr get_area_bg3_bank
;org $A7C7FD : lda #!tiles_2bpp_address+$C00
;org $A7C802 : jsr get_area_bg3_bank

warnpc !bank_a7_freespace_end
