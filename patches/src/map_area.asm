arch snes.cpu
lorom

!bank_82_freespace_start = $82F70F
!bank_82_freespace_end = $82FA80

!tiles_2bpp_address = $C000

!backup_area = $1F62
;!unexplored_gray = #$2108
!unexplored_gray = #$18c6
!unexplored_light_gray = #$35ad
!area_explored_mask = $702600

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

;;; Hijack code that loads area from room header
org $82DE80
    jsl load_area
    jmp $DE89
warnpc $82DE89

;org $828D08
org $828D4B
    jsr pause_start_hook

org $82936A
    jsr pause_end_hook

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
    jsr load_tileset_palette_hook
    nop : nop : nop : nop

org $82E1F7
    jsr palette_clear_hook

;org $82E464
;org $82E55F
;org $82E780
org $82E764
    jsr door_transition_hook

org $82E4A2
    jsr load_target_palette_hook

org $90AB4A
    jsl samus_minimap_flash_hook : nop : nop

; Indicate Samus position on HUD by flashing tile palette 0 instead of palette 7
org $90AB56
    AND #$E3FF     ; was: ORA #$1C00

; Use palette 3 for full ETanks (instead of palette 2)
org $809BDC
    LDX #$2C31     ; was: LDX #$2831


; Use palette 6 (gray/white, same as unexplored tiles) for fixed HUD (e.g. "ENERGY")
org $80988B
dw $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $381D, $381D, $381D, $381D, $381D, $381C,
   $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $3812, $3812, $3823, $3812, $3812, $381E,
   $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $2822, $2822, $2823, $2813, $3814, $381E,
   $380F, $380B, $380C, $380D, $3832, $380F, $3809, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $380F, $3812, $3812, $A824, $2815, $3816, $381E

; Use palette 6 (gray/white, same as unexplored tiles) for HUD digits
org $809DBF : dw $3809, $3800, $3801, $3802, $3803, $3804, $3805, $3806, $3807, $3808
org $809DD3 : dw $3809, $3800, $3801, $3802, $3803, $3804, $3805, $3806, $3807, $3808

; Use palette 0 for full auto reserve
org $80998B             
    dw  $2033, $2046,
        $2047, $2048,
        $A033, $A046

; Use palette 7 for palette blends (FX: water, lava, etc.)
org $89AB62 : STA $7EC03A   ; was: STA $7EC032
org $89AB6A : STA $7EC03C   ; was: STA $7EC034
org $89AB72 : nop : nop : nop : nop   ; was: STA $7EC036
;org $89AB72 : STA $7EC03E   ; was: STA $7EC036

org $89AC05 : STA $7EC23A   ; was: STA $7EC232
org $89AC0D : STA $7EC23C   ; was: STA $7EC234
org $89AC15 : nop : nop : nop : nop   ; was: STA $7EC236
;org $89AC15 : STA $7EC23E   ; was: STA $7EC236
org $89AC1E : nop : nop : nop : nop   ; was: STA $7EC236


;;; Put new code in free space at end of bank $82:
org !bank_82_freespace_start

;;; X = room header pointer
load_area:
    phx
    phy

    ;;; Load the original area number into $079F
    lda $0001,x
    and #$00FF
    sta $079F

    ;;; Load the new area number (for use in map) into $1F5B
    asl
    tay
    lda $E99B, y
    clc
    adc $079D
    tay
    lda $0000, y   ; new/map room area = [[$8F:E99B + (original area) * 2] + room index]
    and #$00FF
    sta $1F5B

    ; mark area as explored (determinines set of valid area maps to cycle through in pause menu):
    jsl $80818E    ; convert map area to bitmask
    lda $05E7      ; load bitmask
    ora !area_explored_mask    ; combine with area explored mask
    sta !area_explored_mask    ; update area explored mask

    lda $1F5B
    ply
    plx
    rtl


PauseRoutineIndex:
	DW $9120, $9142, $9156, $91AB, $9231, $9186, $91D7, $9200	;same as $9110
	DW $9156, switch_map_area, $9200		;fade out / map construction / fade in

pause_start_hook:
    lda $1F5B
    sta !backup_area  ; back up map area
    jsr set_hud_map_colors
    jsr update_pause_map_palette
    ;jsr remove_samus_hud_indicator
    jsl $8085C6  ; save current map explored bits
    ;jsr $8D51  ; run hi-jacked instruction
    inc $0998  ; run hi-jacked instruction
    rts

pause_end_hook:
    lda !backup_area
    sta $1F5B  ; restore map area
    jsl $80858C ; restore map explored bits
    jsr $A2BE  ; run hi-jacked instruction
    rts

check_start_select:
    php
    rep #$30

    lda $8F        ; load newly pressed input
    bit #$2000
    beq .skip      ; if select is not newly pressed, continue as normal

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


switch_map_area:
    jsr next_area
    jsr update_pause_map_palette
    jsl load_bg1_2_tiles
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
    rts

update_pause_map_palette:
    lda $1F5B
    asl
    tax
;    lda area_palettes_unexplored, x

;    ; Set unexplored color to gray: palette 3, color 1
;    lda !unexplored_gray
;    sta $7EC062

    ; Set unexplored gray color: palette 6, color 1
    lda !unexplored_gray
    sta $7EC0C2

    ; Set unexplored light gray color: palette 6, color 2
    lda !unexplored_light_gray
    sta $7EC0C4

    ; Set unexplored white color: palette 6, color 3
    lda #$FFFF
    sta $7EC0C6

;    ; Set color 3 to black (instead of red)
;    lda #$0000
;    sta $7EC066
;    sta $7EC046

    ; Set explored color based on area: palette 2, color 1
    lda area_palettes_explored, x
    sta $7EC042

    ; Set light explored color based on area: palette 2, color 2
    lda area_palettes_explored_light, x
    sta $7EC044

    ; Set explored white: palette 2, color 3
    lda #$7FFF
    sta $7EC046

;    lda !backup_area
;    cmp $1F5B
;    bne .skip_hud_color
;    lda area_palettes_explored, x
;    sta $7EC012  ; set the current area HUD color

    rts

;remove_samus_hud_indicator:
;    ; Remove HUD Samus indicator
;    lda $7EC042
;    sta $7EC03A


area_palettes_explored:
    dw $5c2e  ; Crateria
    dw $0200  ; Brinstar
    dw $0014  ; Norfair
    dw $0210  ; Wrecked Ship
    dw $6163  ; Maridia
    dw $0174  ; Tourian

area_palettes_explored_light:
    dw $7d35  ; Crateria
    dw $2308  ; Brinstar
    dw $1cdd  ; Norfair
    dw $22d8  ; Wrecked Ship
    dw $7E48  ; Maridia
    dw $01fd  ; Tourian


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

load_tileset_palette_hook:
    ; run hi-jacked instruction:
    jsl $80B0FF
    dl $7EC200

    jsr set_hud_map_colors
    jsr load_target_palette

    rts

palette_clear_hook:
    lda $C016  ; preserve unexplored white color (2bpp palette 2, color 3)
    sta $C216

    lda $C032  ; preserve unexplored gray color (2bpp palette 6, color 1)
    sta $C232

    lda $C034  ; preserve unexplored light gray color (2bpp palette 6, color 2)
    sta $C234

    lda $C036  ; preserve unexplored white color (2bpp palette 6, color 2)
    sta $C236

;    lda $C03A  ; preserve pink color for full E-tank energy squares (2-bit palette 7, color 1)
;    sta $C23A

;    lda $C03C  ; preserve white color for full E-tank energy squares (2-bit palette 7, color 2)
;    sta $C23C

;    lda $C03C  ; preserve off-white color in Samus indicator (2-bit palette 7, color 1)
;    sta $C23C

    lda $C014  ; run hi-jacked instruction
    rts

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

    ;lda $7EC032  ; unexplored gray: palette 6, color 1
    lda !unexplored_gray
    sta $7EC232

    ;lda $7EC034  ; unexplored light gray: palette 6, color 2
    lda !unexplored_light_gray
    sta $7EC234

    ;lda $7EC036  ; unexplored white: palette 6, color 3
    lda #$7FFF
    sta $7EC236

    rts

load_target_palette_hook:
    jsr load_target_palette
    lda #$E4A9   ; run hi-jacked instruction
    rts

door_transition_hook:
    jsr set_hud_map_colors
    lda #$0008   ; run hi-jacked instruction
    rts

set_hud_map_colors:
    ; Set colors for HUD map:
    lda $1F5B
    asl
    tax

    ; Set unexplored gray: palette 6, color 1
    lda !unexplored_gray
    sta $7EC032

    ; Set unexplored light gray: palette 6, color 2
    lda !unexplored_light_gray
    sta $7EC034

    ; Set unexplored white: palette 6, color 3
    lda #$7FFF
    sta $7EC036

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
    lda #$48FB
    sta $7EC01A

    ; Set palette 3, color 2 to white color for full E-tank energy squares
    lda #$7FFF
    sta $7EC01C

;    ; water FX: palette 3, color 1 & 2 
;    lda #$0421
;    STA $7EC03A
;;    lda #$1084
;    lda #$0C63
;    STA $7EC03C

;    ; Set unexplored color 3 to pink color for full E-tank energy squares (used for black in vanilla)
;    sta $7EC01E

;    ; Set Samus marker to solid white instead of orange/red (palette 7, colors 1-2)
;    lda #$7FFF
;    sta $7EC03A
;    lda #$6318
;    sta $7EC03C

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


simple_scroll_setup:
    ; Like $829028 but without using Samus position, just midpoints.
    JSR $A0F7    ; Reset pause menu animations
    JSR $9EC4    ; Determine map scroll limits
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

load_bg3_tiles:
    php

    rep #$30
    LDA #$0080
    STA $2115  ; video port control
    LDA #$4000
    STA $2116  ; VRAM (destination) address = $4000

    lda #$1801
    STA $4310  ; DMA control: DMA transfer from CPU to VRAM, incrementing CPU address
    
    lda #!tiles_2bpp_address ; source address
    sta $4312

    ; Set source bank to $E2 + map area:
    lda $1F5B  ; map area (0-5)
    clc
    adc #$00E2
    sta $4314

    lda #$1000
    sta $4315 ; transfer size = $1000 bytes

    sep #$30
    lda #$02
    sta $420B  ; perform DMA transfer on channel 1
    
    plp
    rtl


load_bg3_tiles_door_transition:
    php

    ; source = $E2A200 + map area * $10000
    lda #!tiles_2bpp_address
    sta $05C0
    sep #$30
    lda $1F5B  ; map area (0-5)
    clc
    adc #$E2
    sta $05C2
    rep #$30
    
    ; destination = $4000
    lda #$4000
    sta $05BE

    ; size = $1000
    lda #$1000
    STA $05C3

    LDA #$8000             ;\
    TSB $05BC              ;} Flag door transition VRAM update

.spin:
    LDA $05BC  ;\
    BMI .spin  ;} Wait for door transition VRAM update

    ;lda $1F5B
    ;lda $05F7
    ;sta $09c6

    ; update HUD minimap
    ; jsl $90A91B
    ;lda #$0001
    ;sta $05F7
    ;jsl $809B44

    plp
    rtl

load_bg1_2_tiles:
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

    ; Set source bank to $E2 + map area:
    lda $1F5B  ; map area (0-5)
    clc
    adc #$00E2
    sta $4314

    lda #$4000
    sta $4315 ; transfer size = $2000 bytes

    sep #$30
    lda #$02
    sta $420B  ; perform DMA transfer on channel 1

    plp
    rtl


reload_map_hook:
    phx

    LDA $830002,x  ; run hi-jacked instruction
    BIT #$0040
    beq .skip

    ; clear HUD minimap
    LDX #$0000             ;|
    lda #$381f
.clear_minimap_loop:
    STA $7EC63C,x          ;|
    STA $7EC67C,x          ;} HUD tilemap (1Ah..1Eh, 1..3) = 2C1Fh
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

.skip:
    plx
    LDA $830002,x  ; run hi-jacked instruction
    rtl

warnpc !bank_82_freespace_end

org $82DFB9
    jsl reload_map_hook

; Pause menu: Pink color for full E-tank energy squares in HUD (palette 3, color 1)
org $B6F01A : dw $48FB

; Pause menu: White color for full E-tank energy squares in HUD (palette 3, color 2)
org $B6F01C : dw $7FFF

; Unexplored gray: palette 6, color 1
org $B6F032 : dw !unexplored_gray

; Unexplored light gray: palette 6, color 2
org $B6F034 : dw !unexplored_light_gray

; Unexplored white: palette 6, color 3
org $B6F036 : dw $7FFF

; Patch tile data for button letters. Changing the palettes to 3:
org $858426            
    dw $2CE0, ; A
       $2CE1, ; B
       $2CF7, ; X
       $2CF8, ; Y
       $2CD0, ; Select
       $2CEB, ; L
       $2CF1, ; R
       $2C4E  ; Blank

; Phantoon power-on palette:
org $A7CA77 : dw #$48FB            ; 2bpp palette 2, color 3: pink color for E-tanks (instead of black)
org $A7CA7B : dw !unexplored_gray   ; 2bpp palette 3, color 1: gray color for HUD dotted grid lines

; Load BG3 tiles from source depending on map area:
org $8282F4
    jsl load_bg3_tiles
    rep 13 : nop

; Patch door transition code to always reload BG3 tiles, based on map area:
org $82E46A : beq $1c
org $82E472 : beq $14
org $82E488
    jsl load_bg3_tiles_door_transition
    rep 6 : nop

; Patch pause menu start to load BG1/2 tiles based on map area:
org $828E87 
    jsl load_bg1_2_tiles
    rep 13 : nop

; Patch pause menu start to load BG3 tiles based on map area:
org $828EC7
    jsl load_bg3_tiles
    rep 13 : nop

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


