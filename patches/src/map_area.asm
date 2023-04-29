arch snes.cpu
lorom

!backup_area = $1F62

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
    ;lda #$18C6
    lda #$1CE7
    sta $7EC062

    ; Set explored color based on area:
    lda area_palettes_explored, x
    sta $7EC042

;    ; Set elevator platform color to white (instead of red)
;    lda #$7FFF
;    sta $7EC066    
;    sta $7EC046
    rts

;area_palettes_unexplored:
;    dw $2446  ; Crateria
;    dw $0100  ; Brinstar
;    dw $000A  ; Norfair
;    dw $0108  ; Wrecked Ship
;    dw $2882  ; Maridia
;    dw $1CE7  ; Tourian
;
area_palettes_explored:
;    dw $680F  ; Crateria
    dw $6C50  ; Crateria
    dw $02A0  ; Brinstar
    dw $0019  ; Norfair
    dw $02F6  ; Wrecked Ship
    dw $7D86  ; Maridia
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
    inc $09c6
    rts

warnpc $82F880

