arch snes.cpu
lorom

!backup_area = $1F62
!unexplored_gray = #$2529

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

;org $82952D  ; Draw room select map
;    lda $1F5B
;
;org $829562  ; Draw room select map
;    ldx $1F5B
;
;org $82962B  ; Something map-related (?)
;    lda $1F5B
;
;org $829ED5  ; Something map-related (?)
;    lda $1F5B
;
;org $829F01  ; Something map-related (?)
;    lda $1F5B

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

org $829E38
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


;;; Put new code in free space at end of bank $82:
org $82F70F

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
    ora $7FFE02    ; combine with area explored mask
    sta $7FFE02    ; update area explored mask

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
    jsr update_pause_map_palette
    jsr set_hud_map_colors
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
	jsl $80858C     ;load explored bits for area
    jsl $8293C3		;update area label and construct new area map
    lda #$0080
	jsr $9E27		;set map scroll boundaries and screen starting position
    
    LDA #$0000             ;\
    STA $0723             ;} Screen fade delay = 0

    inc $0727
    rts

update_pause_map_palette:
    lda $1F5B
    asl
    tax
;    lda area_palettes_unexplored, x

    ; Set unexplored color to gray:
    lda !unexplored_gray
    sta $7EC062

    ; Set color 3 to black (instead of red)
    lda #$0000
    sta $7EC066
    sta $7EC046

    ; Set explored color based on area:
    lda area_palettes_explored, x
    sta $7EC042

;    lda !backup_area
;    cmp $1F5B
;    bne .skip_hud_color
;    lda area_palettes_explored, x
;    sta $7EC012  ; set the current area HUD color

    rts

area_palettes_explored:
    dw $6C50  ; Crateria
    dw $02E0  ; Brinstar
    dw $0019  ; Norfair
    dw $02D8  ; Wrecked Ship
    dw $7E44  ; Maridia
    dw $5294  ; Tourian


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
    and $7FFE02     ; test if area is explored
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
    lda $C016  ; preserve pink color for full E-tank energy squares (2-bit palette 2, color 3, black in vanilla)
    sta $C216

;    lda $C03C  ; preserve off-white color in Samus indicator (2-bit palette 7, color 1)
;    sta $C23C

    lda $C014  ; run hi-jacked instruction
    rts

load_target_palette:
    ; Prevent HUD map colors from gradually changing (e.g. to blue/pink) during door transition:
    lda $7EC01A  ; unexplored gray
    sta $7EC21A

    lda $7EC012  ; explored color
    sta $7EC212

    lda $7EC016  ; pink color for full E-tank energy squares (using color 3, used for black in vanilla)
    sta $7EC216

 ;   lda $7EC03A
 ;   sta $7EC23A

;    lda $7EC03C
;    sta $7EC23C

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

    ; Set unexplored color to gray:
    lda !unexplored_gray
    sta $7EC01A

    ; Set explored color based on area:
    lda.l area_palettes_explored, x
    sta $7EC012

    ; Set explored color 3 to pink color for full E-tank energy squares (used for black in vanilla)
    lda #$48FB
    sta $7EC016
;    sta $7EC01E

;    ; Set unexplored color 3 to pink color for full E-tank energy squares (used for black in vanilla)
;    sta $7EC01E

;    ; Set Samus marker to solid white instead of orange/red (palette 7, colors 1-2)
;    lda #$7FFF
;    sta $7EC03A
;    lda #$6318
;    sta $7EC03C

    rts

warnpc $82F900

;; Pause menu: Samus indicator in HUD (palette 7, colors 1-2)
;org $B6F03A
;    dw $7FFF
;    dw $6318

; Pause menu: Pink color for full E-tank energy squares in HUD (palette 2, color 3)
org $B6F016
    dw $48FB

; Pause menu: Gray color for unexplored tiles in HUD (palette 3, color 1)
org $B6F01A
    dw !unexplored_gray

; Pause menu: Black color for 4bpp color 3 (palette 2 - explored)
org $B6F046
    dw $0000

; Pause menu: Black color for 4bpp color 3 (palette 3 - unexplored)
org $B6F066
    dw $0000
