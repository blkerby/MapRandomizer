; Objectives pause menu screen based on VARIA Randomizer's patch by theonlydude/ouiche
;
; Simplified the objectives implementation and ported much of the code to reside in bank 85.
;
; Randomizer defines the objectives @ $B6F200 with the following format:
;
; - Each line terminates with a word value of $8000
; - Max length of line is 30 displayable characters
; - Valid characters defined in tables/pause_menu_objectives_chars.tbl. They must be converted to 
;   the corresponding word value in the .tbl before writing to ROM
; - Line count is 18 and must all be defined (even if blank)
;
; !check_char is reserved for objective checkmark locations.
;
; Stag Shot

lorom
arch 65816

math pri on

incsrc "constants.asm"

!bank_82_free_space_start = $82FF80
!bank_82_free_space_end = $82FFFC

!bank_85_free_space_start = $859B20
!bank_85_free_space_end = $859FF0

!bank_B6_free_space_start = $B6F200
!bank_B6_free_space_end = $B6F660

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; HIJACKS

;;; new pause index func list
org $82910A
    JSR (new_pause_index_func_list,x)
    
;;; simplify unpausing
org $82932B
    JSL display_unpause : nop : nop : nop

;;; update BG2 buttons
org $82A62D
    JSL set_bg2_equipment_screen : nop : nop

org $82A79B
    JSL set_bg2_map_screen : nop : nop

;;; keep 'MAP' left button visible on map screen by keeping palette 2 instead of palette 5 (grey one)
org $82A820
    ORA #$1000

org $82A83E
    ORA #$1000

;;; update glowing sprite around L/R pointer
org $82C1E6
    dw glowing_LR_animation
    
;;; new function to check for L/R button pressed
org $82A505
    JML check_l_r_pressed : nop : nop

;;; Replace pause screen button label palettes functions
org $82A61D
    JSR (new_pause_palettes_func_list,x)
    
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Pause stuff

!n_lines = #$0012

;;; character conversion table
table "tables/pause_menu_objectives_chars.tbl",RTL

;;; pause state
!pause_index = $0727

;;; fast_pause_menu
!fast_pause_menu = $82fffc      ; must match address in patch.rs

;;; pause index values
!pause_index_map_screen = #$0000
!pause_index_equipment_screen = #$0001
!pause_index_map2equip_fading_out = #$0002
!pause_index_map2equip_load_equip = #$0003
!pause_index_map2equip_fading_in = #$0004
!pause_index_equip2map_fading_out = #$0005
!pause_index_equip2map_load_map = #$0006
!pause_index_equip2map_fading_in = #$0007
;;; new screen (skip 3 indices used by map patch):
!pause_index_objective_screen = #$000B
!pause_index_map2obj_fading_out = #$000C
!pause_index_map2obj_load_obj = #$000D
!pause_index_map2obj_fading_in = #$000E
!pause_index_obj2map_fading_out = #$000F
!pause_index_obj2map_load_map = #$0010
!pause_index_obj2map_fading_in = #$0011

;;; pause screen button label mode
!pause_screen_button_mode = $0753
!pause_screen_button_map = #$0000     ; Map screen (SAMUS on the right, OBJ on the left)
!pause_screen_button_nothing = #$0001 ; Unpausing (nothing)
!pause_screen_button_equip = #$0002   ; Equipment screen (MAP on the left)
;;; new button mode:
!pause_screen_button_obj = #$0003     ; Objective screen (MAP on the right)

;;; Pause screen mode
!pause_screen_mode = $0763

;;; pause screen mode values
!pause_screen_mode_map = #$0000
!pause_screen_mode_equip = #$0001
;;; new mode:
!pause_screen_mode_obj = #$0002

;;; button stuff
!held_buttons = $05E1
!newly_pressed_buttons = $8F
!l_button = #$0020
!r_button = #$0010
!light_up_l_button  = #$0001
!light_up_r_button  = #$0002

;; dynamic objective text: BG1 tilemap in RAM
!BG1_tilemap = $7E3800
;; rows [5, 23] of screen
!BG1_tilemap_size = $4c0

!line_size #= 32*2
!check_char = #$28DD ; "-" = char to replace with check (1st occurrence)

;; RAM
;; current first objective displayed
!obj_index = $073d

;;; simple helper to instant DMA gfx from a static long source address to VRAM
;;; usable during blank screen only
macro gfxDMA(src, dstVRAM, size)
    PHP
    SEP #$30
    LDA.b #(<dstVRAM>&$ff) : STA $2116  ;| VRAM Address Registers (Low)
    LDA.b #(<dstVRAM>>>8) : STA $2117   ;| VRAM Address Registers (High)
    LDA.b #$80 : STA $2115    ;} Video Port Control Register - Set VRAM transfer mode to word-access, increment by 1.
                  ;    0x80 == 0b10000000 => i---ffrr => i=1 (increment when $2119 is accessed),
                  ;    ff=0 (full graphic ??), rr=0 (increment by 2 bytes)
    JSL $8091A9 ; Set up a DMA transfer
    db $01,$01      ; hard-coded channel 1, options = $01
    db $18          ; DMA target = VRAM
    dl <src>
    dw <size>
    LDA.b #$02 : STA $420B   ; start transfer
    PLP
endmacro

;;; helper to DMA load from static long address (*not RAM*) to static long address (RAM)
macro loadRamDMA(src, dstRAMl, size)
    PHP
    SEP #$30
    ;; write RAM address to proper registers
    LDA.b #(<dstRAMl>&$ff) : STA $2181
    LDA.b #((<dstRAMl>&$ff00)>>8) : STA $2182
    LDA.b #(<dstRAMl>>>16) : STA $2183
    ;; set up DMA transfer
    JSL $8091A9
    db $01,$00      ; hard-coded channel 1, options = $00
    db $80          ; DMA target = WRAM
    dl <src>
    dw <size>
    LDA #$02 : STA $420B   ; start transfer
    PLP
endmacro

;;; simple helper to queue DMA gfx from a static long source address to VRAM
;;; usable at any time, uses X
macro queueGfxDMA(src, dstVRAM, size)
    LDX $0330
    LDA.w #<size> : STA.b $D0,x
    INX : INX
    LDA.W #(<src>&$ffff) : STA.b $D0,x
    INX : INX
    SEP #$20
    LDA.b #(<src>>>16) : STA.b $D0,x
    REP #$20
    INX
    LDA.w #<dstVRAM> : STA.b $D0,x
    INX : INX
    STX $0330
endmacro

;;; call bank 82 funcs from bank 85
macro callBank82Func(func)
    PHX
    LDX.w #<func>
    DEX
    JSL hook_85_to_82
    PLX
endmacro

org !bank_82_free_space_start
;;; glowing sprites around L/R
glowing_LR_animation:
    dw $002A, $002A, $002A, $002A

new_pause_palettes_func_list:
    dw $A796, $A6DF, $A628, update_palette_objective_screen

update_palette_objective_screen:
    JSL update_palette_objective_screen_85
    RTS

hook_85_to_82:          ; X = local func
    PEA ret_long-1
    PHX                 ; stack hack
    RTS                 ; to call X
ret_long:
    RTL

new_pause_index_func_list: ; expanded pause index func list
dw $9120, $9142, $9156, $91AB, $9231, $9186, $91D7, $9200 ; stock 0-7
dw $9156, $F816, $9200 ; map_area.asm 8-0A
dw func_pause_index_objective_screen, func_pause_index_map2obj_fading_out, func_pause_index_map2obj_load_obj ; 0B-0D
dw func_pause_index_map2obj_fading_in, func_pause_index_obj2map_fading_out, $91D7, $9200 ; 0E-11

; bank 82 to 85 calls for indexes 0B-0F
func_pause_index_objective_screen:
    JSL func_objective_screen
    RTS
    
func_pause_index_map2obj_fading_out:
    JSL func_map2obj_fading_out
    RTS

func_pause_index_map2obj_load_obj:
    JSL func_map2obj_load_obj
    RTS

func_pause_index_map2obj_fading_in:
    JSL func_map2obj_fading_in
    RTS

func_pause_index_obj2map_fading_out:
    JSL func_obj2map_fading_out
    RTS

print "82 end: ", PC
warnpc !bank_82_free_space_end

;;; continue in bank 85 for obj screen management code
org !bank_85_free_space_start

;;; load objective screen title from ROM to RAM
load_obj_tilemap:
    %loadRamDMA(obj_bg1_tilemap, !BG1_tilemap+20, 24)
    RTL

;;; update RAM tilemap with objectives text, line by line
!tmp_tile_offset = $12

update_objs:
    PHB
    PHK
    PLB
    LDA.w #!line_size+2
    STA !tmp_tile_offset
    STZ !obj_index
    LDY #$ffff
    
.draw_obj_text
    INY
    CPY !n_lines
    BEQ check_objs
    LDX !tmp_tile_offset
    
.draw_obj_loop
    PHX
    LDX !obj_index
    LDA obj_txt_ptrs, x
    INX : INX
    STX !obj_index
    PLX
    CMP #$8000
    BEQ .pad_rest_of_line
    STA !BG1_tilemap, x
    INX : INX
    BRA .draw_obj_loop

.pad_rest_of_line
    JSR pad_0
    BRA .draw_obj_text

;;; pad with 0s until end of line
pad_0:
    TXA
    AND #$003F
    BEQ .end
    LDA #$280E : STA.l !BG1_tilemap,x
    INX : INX
    BRA pad_0
    
.end:
    INX : INX
    STX !tmp_tile_offset
    RTS
    
check_objs:
;;; check objectives and add check marks
    LDY.w #!line_size*2            ; start of 1st line
    LDA !objectives_num : AND $7FFF
    STA !tmp_tile_offset           ; # objectives
    BEQ .exit
    LDX #$0000

.obj_check_lp
    PHX
    JSR check_objective
    BEQ .nocheck

    ; find the next check character and mark it green
    TYX
    JSR find_next_check
    LDA #$250B                     ; check mark (green)
    STA !BG1_tilemap, x        
    BRA .next

.nocheck:
    ; find the next check character and skip marking it
    TYX                            ; X <- position in tilemap 
    JSR find_next_check

.next:    
    INX : INX                      ; advance to next position in tilemap
    TXY
    PLX                            ; X <- objective number
    INX
    CPX !tmp_tile_offset
    BEQ .exit
    BRA .obj_check_lp

.exit
    PLB
    RTL

check_objective: ; X = index
    PHX
    TXA
    ASL
    TAX
    LDA.w #$007E
    STA.b $02
    LDA.l !objectives_addrs, X
    STA.b $00
    LDA.l !objectives_bitmasks, X
    STA.b $04
    LDA.b [$00]
    PLX
    BIT.b $04
    RTS

find_next_check:
    LDA !BG1_tilemap, x
    CMP !check_char                ; tile to switch?
    BEQ .found
    INX : INX
    BRA find_next_check
.found:
    RTS
    
;;; direct DMA of BG1 tilemap to VRAM
blit_objs:
    %gfxDMA(!BG1_tilemap, $30a0, !BG1_tilemap_size)
    RTL

;;; DMA tilemap each frame
queue_obj_tilemap:
    %queueGfxDMA(!BG1_tilemap, $30a0, !BG1_tilemap_size)
    RTL

;;; check if up/down press, and if applicable, scroll (play sfx for scroll ok/ko)
obj_scroll:
    RTL

;;; check for L or R input and update pause_index && pause_screen_button_mode
check_l_r_pressed:
    PHP
    REP #$30
    LDA !held_buttons
    BIT !l_button
    BNE .press_L
    BIT !r_button
    BNE .press_R
    BRA .end

.press_R:
    LDA !pause_screen_button_mode
    CMP !pause_screen_button_equip  ; if already equipment screen => end
    BEQ .end
    ;; common actions
    LDA $C10A   ; $82:C10A         db 05,00
    STA $0729   ; Frames to flash L/R/start button on pause screen
    LDA !light_up_r_button
    STA $0751   ; $0751: Which button lights up for $0729 frames when changing screens from pause screen 

    LDA !pause_screen_button_mode
    CMP !pause_screen_button_obj
    BEQ .move_to_map_from_obj

.move_to_equip_from_map:
    LDA !pause_index_map2equip_fading_out
    STA !pause_index
    LDA !pause_screen_button_equip
    STA !pause_screen_button_mode
    BRA .play_sound

.move_to_map_from_obj:
    LDA !pause_index_obj2map_fading_out
    STA !pause_index
    LDA !pause_screen_button_map
    STA !pause_screen_button_mode        ; pause_screen_button_mode set to pause_screen_button_equip
    BRA .play_sound

.press_L:
    LDA !pause_screen_button_mode        ; pause_screen_button_mode, 00 == map screen
    CMP !pause_screen_button_obj
    BEQ .end                             ; if already on objective screen => end
    ;; common actions
    LDA $C10A                            ; $82:C10A         db 05,00
    STA $0729                            ; frames to flash L/R/start button on pause screen
    LDA !light_up_l_button
    STA $0751

    LDA !pause_screen_button_mode
    CMP !pause_screen_button_map
    BEQ .move_to_obj_from_map            ; if on map screen and L pressed => objective screen

.move_to_map_from_equip:
    LDA !pause_index_equip2map_fading_out
    STA !pause_index
    STZ !pause_screen_button_mode        ; pause_screen_button_mode set to pause_screen_button_map
    BRA .play_sound

.move_to_obj_from_map:
    LDA !pause_index_map2obj_fading_out
    STA !pause_index
    LDA !pause_screen_button_obj
    STA !pause_screen_button_mode
    
.play_sound:
    %callBank82Func($A615)               ; $A615: Set pause screen buttons label palettes to show/hide them
    LDA #$0038                           ;\
    JSL $809049                          ;} Queue sound 38h, sound library 1, max queued sounds allowed = 6 (menu option selected)

.end
    PLP
    JML $82A59A

;;; unpause
display_unpause:
    LDA !pause_screen_mode
    CMP !pause_screen_mode_equip
    BEQ .equip
    CMP !pause_screen_mode_obj
    BEQ .objective
; map
    JSL $82BB30               ; Display map elevator destinations
    JSL $82B672               ; Draw map icons
    %callBank82Func($B9C8)    ; Draw Samus indicator
    BRA .objective
.equip:
    %callBank82Func($B267)    ; Draw item selector
    %callBank82Func($B2A2)    ; Display reserve tank amount
.objective:
    %callBank82Func($A56D)    ; Updates the flashing buttons when you change pause screens
    LDA !fast_pause_menu : AND #$8000
    BNE .fast
    JSL $808924               ; Handle fading out
    RTL
.fast
    JSL fast_fadeout
    RTL

;;; buttons addresses in BG2
!left_button_top     = $7E364A
!left_button_bottom  = $7E368A
!right_button_top    = $7E366C
!right_button_bottom = $7E36AC

;;; replace 'MAP' with 'OBJ' in left BG2, put back 'SAMUS' in right BG2
set_bg2_map_screen:
    PHB
    PHK
    PLB
    LDY #$000A
    LDX #$0000
.left_loop_top:
    LDA obj_top,x
    STA !left_button_top,x
    INX : INX
    DEY : DEY
    BNE .left_loop_top

    LDY #$000A
    LDX #$0000
.left_loop_bottom:
    LDA obj_bottom,x
    STA !left_button_bottom,x
    INX : INX
    DEY : DEY
    BNE .left_loop_bottom

    LDY #$000A
    LDX #$0000
.right_loop_top:
    LDA samus_top,x
    STA !right_button_top,x
    INX : INX
    DEY : DEY
    BNE .right_loop_top

    LDY #$000A
    LDX #$0000
.right_loop_bottom:
    LDA samus_bottom,x
    STA !right_button_bottom,x
    INX : INX
    DEY : DEY
    BNE .right_loop_bottom
    LDY #$000A                     ; vanilla code
    LDX #$0000
    PLB
    RTL

;;; put back 'MAP' in BG2 left
set_bg2_equipment_screen:
    PHB
    PHK
    PLB
    LDY #$000A
    LDX #$0000
.loop_top:
    LDA map_top,x
    STA !left_button_top,x
    INX : INX
    DEY : DEY
    BNE .loop_top

    LDY #$000A
    LDX #$0000
.loop_bottom:
    LDA map_bottom,x
    STA !left_button_bottom,x
    INX : INX
    DEY : DEY
    BNE .loop_bottom
    LDY #$000A                     ; vanilla code
    LDX #$0000
    PLB
    RTL

;;; obj:   left: grey (obj), right: MAP
;;; map:   left: OBJ,    right: samus
;;; equip: left: map,    right: grey (samus)

;;; obj/map/samus buttons tiles
obj_top:
    dw $28E4, $290C, $290D, $290E, $28E8
obj_bottom:
    dw $28F4, $291C, $291D, $291E, $28F8

map_top:
    dw $28E4, $28E5, $28E6, $28E7, $28E8
map_bottom:
    dw $28F4, $28F5, $28F6, $28F7, $28F8
 
samus_top:
    dw $28E9, $28EA, $28EB, $28EC, $28ED
samus_bottom:
    dw $28F9, $28FA, $28FB, $28FC, $28FD

update_palette_objective_screen_85:
    PHP
    REP #$30
    PHB
    PHK
    PLB

    LDY #$000A
    LDX #$0000
.loop_top
    LDA map_top,x
    STA !right_button_top,x
    INX : INX
    DEY : DEY
    BNE .loop_top

    LDY #$000A
    LDX #$0000
.loop_bottom
    LDA map_bottom,x
    STA !right_button_bottom,x
    INX : INX
    DEY : DEY
    BNE .loop_bottom

    LDY #$000A
    LDX #$0000
.loop_top2
    LDA !right_button_top,x
    AND #$E3FF
    ORA #$1000
    STA !right_button_top,x        ; Set tilemap palette indices at $7E:364A..53 to 5 (top of MAP)
    INX : INX
    DEY : DEY
    BNE .loop_top2

    LDY #$000A
    LDX #$0000
.loop_bottom2
    LDA !right_button_bottom,x
    AND #$E3FF
    ORA #$1000
    STA !right_button_bottom,x     ; Set tilemap palette indices at $7E:368A..93 to 5 (bottom of MAP)
    INX : INX
    DEY : DEY
    BNE .loop_bottom2

    LDY #$000A
    LDX #$0000
.loop_top3
    LDA !left_button_top,x
    AND #$E3FF
    ORA #$1400
    STA !left_button_top,x         ; Set tilemap palette indices at $7E:364A..53 to 5 (grey)
    INX : INX
    DEY : DEY
    BNE .loop_top3

    LDY #$000A
    LDX #$0000
.loop_bottom3
    LDA !left_button_bottom,x
    AND #$E3FF
    ORA #$1400
    STA !left_button_bottom,x      ; Set tilemap palette indices at $7E:368A..93 to 5 (grey)
    INX : INX
    DEY : DEY
    BNE .loop_bottom3
    PLB
    PLP
    RTL

func_objective_screen:
    %callBank82Func($A505)         ; Checks for L or R input during pause screens
    %callBank82Func($A5B7)         ; Checks for start input during pause screen
    ; disabled for now since content is static
    ;JSL obj_scroll
    ;JSL update_objs
    ;JSL queue_obj_tilemap
    LDA !pause_screen_mode_obj
    STA !pause_screen_mode         ; Pause screen mode = objective screen
    RTL

func_map2obj_fading_out:
    %callBank82Func($A56D)         ; Updates the flashing buttons when you change pause screens
    LDA !fast_pause_menu : AND #$8000
    BNE .fast
    JSL $808924                    ; Handle fading out
    BRA .next
.fast
    JSL fast_fadeout

.next
    SEP #$20
    LDA $51                        ;\
    CMP #$80                       ;} If not finished fading out: return
    BNE .end                       ;/
    JSL $80834B                    ; Enable NMI
    REP #$20
    STZ $0723                      ; Screen fade delay = 0
    STZ $0725                      ; Screen fade counter = 0
    INC !pause_index               ; Pause index = 6 (equipment screen to map screen - load map screen)

    ;; save RAM 3800-3fff to 5000-57ff
    LDA #$7ff
    LDX #$3800
    LDY #$5000
    MVN $7E, $7E

.end:
    RTL

func_map2obj_load_obj:
    REP #$30
    ;; backup map's scroll
    LDA $B1
    STA $BD                        ; BG4 X scroll = [BG1 X scroll]
    LDA $B3
    STA $BF                        ; BG4 Y scroll = [BG1 Y scroll]
    ;; no scroll
    STZ $B1                        ; BG1 X scroll = 0
    STZ $B3                        ; BG1 Y scroll = 0

    STZ !obj_index
    JSL load_obj_tilemap
    JSL update_objs
    JSL blit_objs
    LDA !pause_screen_mode_obj
    STA !pause_screen_mode         ; Pause screen mode = objective screen
    %callBank82Func($A615)         ; Set pause screen button label palettes
    STZ $073F
    LDA $C10C
    STA $072B                      ; $072B = Fh
    LDA #$0001
    STA $0723                      ; Screen fade delay = 1
    STA $0725                      ; Screen fade counter = 1
    INC !pause_index               ; Pause index = B (map screen to objective screen - fading in)
    RTL

func_map2obj_fading_in:
    LDA !pause_screen_mode_obj
    STA !pause_screen_mode         ; Pause screen mode = objective screen
    LDA !fast_pause_menu : AND #$8000
    BNE .fast
    JSL $80894D                    ; Handle fading in
    BRA .next
.fast
    JSL fast_fadein

.next
    SEP #$20
    LDA $51                        ;\
    CMP #$0F                       ; If not finished fading in: return
    BNE .end                       ;/
    REP #$20
    STZ $0723                      ; Screen fade delay = 0
    STZ $0725                      ; Screen fade counter = 0
    LDA !pause_screen_button_obj
    STA !pause_screen_button_mode
    LDA !pause_index_objective_screen ; index = objective
    STA !pause_index
.end:
    RTL

func_obj2map_fading_out:
    ;; fade out to map
    %callBank82Func($A56D)         ; Updates the flashing buttons when you change pause screens
    LDA !fast_pause_menu : AND #$8000
    BNE .fast
    JSL $808924                    ; Handle fading out
    BRA .next
.fast
    JSL fast_fadeout

.next
    SEP #$20
    LDA $51                        ;\
    CMP #$80                       ; If not finished fading out: return
    BNE .end                       ;/
    JSL $80834B                    ; Enable NMI
    REP #$20
    STZ $0723                      ; Screen fade delay = 0
    STZ $0725                      ; Screen fade counter = 0
    INC !pause_index               ; Pause index = D (obj screen to map screen - load map screen)

    ;; restore RAM 3800-3fff from 5000-57ff (needed for equipment screen)
    LDA #$7FF
    LDX #$5000
    LDY #$3800
    MVN $7E, $7E

.end:
    RTL

; Variation of $808924 that goes twice as fast:
fast_fadeout:
    PHP
    SEP #$30
    LDA $51
    AND #$0F                       ; If (brightness) = 0: return
    BEQ .done
    DEC A
    BEQ .force_blank
    DEC A
    BNE .store
.force_blank:
    LDA #$80
.store:
    STA $51
.done:
    PLP
    RTL

; Variation of $80894D that goes twice as fast:
fast_fadein:
    PHP
    SEP #$30

    LDA $51
    INC A
    AND #$0F                       ; If brightness is not max:
    BEQ .done
    STA $51                        ; Increment brightness (disable forced blank)

    INC A
    AND #$0F                       ; If brightness is not max:
    BEQ .done
    STA $51                        ; Increment brightness (disable forced blank)
.done:
    PLP
    RTL

obj_bg1_tilemap:
    ;; line 0 : objectives screen "window title"
    dw $280F, "OBJECTIVES", $280F

print "85 end: ", pc
warnpc !bank_85_free_space_end

org !bank_B6_free_space_start
obj_txt_ptrs:
;; max size for single screen: (30 char dw + terminating dw) * 18 lines = 1116 bytes

warnpc !bank_B6_free_space_end

; objective screen tiles
; 'Q'
org $b69a00
    db $7C, $00, $C6, $00, $C6, $00, $C6, $00, $DA, $00, $CC, $00, $76, $00, $00, $00
    db $FF, $7C, $FF, $C6, $FF, $C6, $FF, $C6, $FF, $DA, $FF, $CC, $FF, $76, $FF, $00

; '.'
org $b69b60
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $18, $00, $18, $00, $00, $00
    db $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $18, $FF, $18, $FF, $00

; '/'
org $b69b80
    db $03, $00, $06, $00, $0C, $00, $18, $00, $30, $00, $60, $00, $C0, $00, $00, $00
    db $FF, $03, $FF, $06, $FF, $0C, $FF, $18, $FF, $30, $FF, $60, $FF, $C0, $FF, $00

; '!'
org $b69be0
    db $18, $00, $18, $00, $18, $00, $18, $00, $00, $00, $18, $00, $18, $00, $00, $00
    db $FF, $18, $FF, $18, $FF, $18, $FF, $18, $FF, $00, $FF, $18, $FF, $18, $FF, $00

; ':'
org $b6a0c0
    db $00, $00, $18, $00, $18, $00, $00, $00, $00, $00, $18, $00, $18, $00, $00, $00
    db $FF, $00, $FF, $18, $FF, $18, $FF, $00, $FF, $00, $FF, $18, $FF, $18, $FF, $00

; check mark
org $b6a160
    db $01, $00, $03, $00, $06, $00, $8C, $00, $D8, $00, $70, $00, $20, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

; top of 'OBJ' button
org $b6a180
    db $00, $FF, $00, $00, $FF, $FF, $FF, $FF, $F8, $F8, $F0, $F0, $F2, $F2, $F2, $F2
    db $FF, $FF, $FF, $FF, $00, $FF, $00, $FF, $07, $F8, $0F, $F0, $0D, $F2, $0D, $F2
    db $00, $FF, $00, $00, $FF, $FF, $FF, $FF, $C1, $C1, $4C, $4C, $4C, $4C, $41, $41
    db $FF, $FF, $FF, $FF, $00, $FF, $00, $FF, $3E, $C1, $B3, $4C, $B3, $4C, $BE, $41
    db $00, $FF, $00, $00, $FF, $FF, $FF, $FF, $F3, $F3, $F3, $F3, $F3, $F3, $F3, $F3
    db $FF, $FF, $FF, $FF, $00, $FF, $00, $FF, $0C, $F3, $0C, $F3, $0C, $F3, $0C, $F3

; bottom of 'OBJ' button
org $b6a380
    db $F2, $F2, $F2, $F2, $F0, $F0, $F8, $F8, $FF, $FF, $00, $FF, $00, $00, $00, $FF
    db $0D, $F2, $0D, $F2, $0F, $F0, $07, $F8, $00, $FF, $FF, $00, $FF, $FF, $FF, $FF
    db $41, $41, $4C, $4C, $4C, $4C, $C1, $C1, $FF, $FF, $00, $FF, $00, $00, $00, $FF
    db $BE, $41, $B3, $4C, $B3, $4C, $3E, $C1, $00, $FF, $FF, $00, $FF, $FF, $FF, $FF
    db $93, $93, $93, $93, $83, $83, $C7, $C7, $FF, $FF, $00, $FF, $00, $00, $00, $FF
    db $6C, $93, $6C, $93, $7C, $83, $38, $C7, $00, $FF, $FF, $00, $FF, $FF, $FF, $FF

; 0-9 top-justified
org $b6ac00
db $7C, $00, $C6, $00, $C6, $00, $C6, $00, $C6, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $C6, $FF, $C6, $FF, $C6, $FF, $C6, $FF, $7C, $FF, $00
db $1C, $00, $3C, $00, $6C, $00, $0C, $00, $0C, $00, $0C, $00, $0C, $00, $00, $00
db $FF, $1C, $FF, $3C, $FF, $6C, $FF, $0C, $FF, $0C, $FF, $0C, $FF, $0C, $FF, $00
db $7C, $00, $C6, $00, $06, $00, $7C, $00, $C0, $00, $C0, $00, $FE, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $06, $FF, $7C, $FF, $C0, $FF, $C0, $FF, $FE, $FF, $00
db $7C, $00, $C6, $00, $06, $00, $1C, $00, $06, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $06, $FF, $1C, $FF, $06, $FF, $C6, $FF, $7C, $FF, $00
db $1C, $00, $3C, $00, $6C, $00, $CC, $00, $FE, $00, $0C, $00, $0C, $00, $00, $00
db $FF, $1C, $FF, $3C, $FF, $6C, $FF, $CC, $FF, $FE, $FF, $0C, $FF, $0C, $FF, $00
db $FE, $00, $C0, $00, $FC, $00, $06, $00, $06, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $FE, $FF, $C0, $FF, $FC, $FF, $06, $FF, $06, $FF, $C6, $FF, $7C, $FF, $00
db $7C, $00, $C6, $00, $C0, $00, $FC, $00, $C6, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $C0, $FF, $FC, $FF, $C6, $FF, $C6, $FF, $7C, $FF, $00
db $FE, $00, $06, $00, $0C, $00, $18, $00, $30, $00, $60, $00, $C0, $00, $00, $00
db $FF, $FE, $FF, $06, $FF, $0C, $FF, $18, $FF, $30, $FF, $60, $FF, $C0, $FF, $00
db $7C, $00, $C6, $00, $C6, $00, $7C, $00, $C6, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $C6, $FF, $7C, $FF, $C6, $FF, $C6, $FF, $7C, $FF, $00
db $7C, $00, $C6, $00, $C6, $00, $7E, $00, $06, $00, $C6, $00, $7C, $00, $00, $00
db $FF, $7C, $FF, $C6, $FF, $C6, $FF, $7E, $FF, $06, $FF, $C6, $FF, $7C, $FF, $00
