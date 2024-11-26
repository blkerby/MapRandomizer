; New Reserve HUD Style
; Each reserve is represented with a two tile wide half tile high bar that shows energy progress per pixel

; Info for portability:
; - This patch uses four tiles to show the reserves. Adjust the "rhud" variables to choose which tiles to use.
; - There are several hooks in various places to help update the tiles in VRAM at right times.
; - $0A1A in RAM is used to store the previous reserve health, used to avoid unnecessarily updating reserve HUD.
; - Direct Page $C1 to $CE is used to store function arguments.

lorom

!bank_82_free_space_start = $82FF30
!bank_82_free_space_end = $82FF80

!bank_84_free_space_start = $84F490
!bank_84_free_space_end = $84F4A0

; Definitions
!samus_max_reserves = $09D4
!samus_reserves = $09D6
!samus_previous_reserves = $0A1A ; Previously unused

; Stores 16 bytes of the special tile while painting. (Bank 7E)
!special_tile_loc = $F4A0

; Reserve HUD tile info
!rhud_i_bl = #$2033 ; Previously end of reserve arrow
!rhud_i_br = #$2046 ; Previously start of reserve arrow
!rhud_i_tl = #$204C
!rhud_i_tr = #$204D

; Reserve HUD tile addresses (VRAM)
!rhud_v_bl = #$0198 ; Previously end of reserve arrow
!rhud_v_br = #$0230 ; Previously start of reserve arrow
!rhud_v_tl = #$0260
!rhud_v_tr = #$0268

; Variables helpers
!tile_info = $C1         ; The tile that should be drawn
!vram_target = $C3       ; The address in VRAM that the tile should be written to
!affect_right_tile = $C5 ; Bool: If set, draw the right column of the bars
!affect_upper_bar = $C7  ; Bool: If set, the top bar of the tile is affected
!healthCheck_Lower = $C9 ; Is compared with reserves and max reserves ...
!healthCheck_Upper = $CB ; ... to determine what to draw
!special_helper = $CD    ; Used to store various info to help draw the special tile correct



; Don't call broken function that writes garbage to reserve HUD
org $82AED9
    NOP #3

; Don't clear the bar reserve tiles
org $82AF36
    NOP #4
    NOP #4
    NOP #4
    NOP #4

; Hook: At "Samus previous health = 0" in HUD init
org $809AE4
    JSR HOOK_HUD_INIT

; Here's where the regular reserve HUD tiles are set; jump to custom draw function instead
org $809B4E
    JSR FUNCTION_DRAW_RESERVE_HUD
    JMP $9B8B ; Jump to drawing e-tanks

org $80D340
TABLE_NEW_TILES: ; Order matters!
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $7F, $3F, $7F, $3F, $7F, $00, $00 ; One Reserve | Empty | Left
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $FC, $FC, $FC, $FC, $FC, $00, $00 ; One Reserve | Empty | Right
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $7F, $3F, $40, $3F, $40, $00, $00 ; One Reserve | Full | Left
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $FC, $FC, $00, $FC, $00, $00, $00 ; One Reserve | Full | Right
    db $00, $7F, $3F, $7F, $3F, $7F, $00, $00, $00, $7F, $3F, $7F, $3F, $7F, $00, $00 ; Two Reserve | Empty/Empty | Left
    db $00, $FC, $FC, $FC, $FC, $FC, $00, $00, $00, $FC, $FC, $FC, $FC, $FC, $00, $00 ; Two Reserve | Empty/Empty | Right
    db $00, $7F, $3F, $7F, $3F, $7F, $00, $00, $00, $7F, $3F, $40, $3F, $40, $00, $00 ; Two Reserve | Full/Empty | Left
    db $00, $FC, $FC, $FC, $FC, $FC, $00, $00, $00, $FC, $FC, $00, $FC, $00, $00, $00 ; Two Reserve | Full/Empty | Right
    db $00, $7F, $3F, $40, $3F, $40, $00, $00, $00, $7F, $3F, $40, $3F, $40, $00, $00 ; Two Reserve | Full/Full | Left
    db $00, $FC, $FC, $00, $FC, $00, $00, $00, $00, $FC, $FC, $00, $FC, $00, $00, $00 ; Two Reserve | Full/Full | Right
FUNCTION_DRAW_RESERVE_HUD:
    LDA $09C0
    CMP #$0001
    BNE FDRH_CHECK_PREV
FDRH_DRAW_AUTO_TEXT:
    LDY #$998B
    LDA !samus_reserves
    BNE $03
    LDY #$9997
    LDA $0004,y ; AU
    STA $7EC698 ; Left
    LDA $0006,y ; TO
    STA $7EC69A ; Right
FDRH_CHECK_PREV:
    LDA !samus_reserves
    CMP !samus_previous_reserves
    STA !samus_previous_reserves
    BNE FDRH_DRAW_TILES ; Only update if reserve energy is different from last frame
    RTS
FDRH_DRAW_TILES:
    ; Bottom left
    LDA !rhud_i_bl : STA !tile_info
    JSR FUNCTION_GET_BG3_BASE : CLC : ADC !rhud_v_bl : STA !vram_target
    LDA #$0000 : STA !affect_right_tile
    LDA #$0000 : STA !healthCheck_Lower  ; 0
    LDA #$0064 : STA !healthCheck_Upper  ; 100
    JSR FUNCTION_DRAW_TILE
    STA $7EC658
    ; Bottom right
    LDA !rhud_i_br : STA !tile_info
    JSR FUNCTION_GET_BG3_BASE : CLC : ADC !rhud_v_br : STA !vram_target
    LDA #$0001 : STA !affect_right_tile
    LDA #$0032 : STA !healthCheck_Lower  ; 50
    LDA #$0096 : STA !healthCheck_Upper  ; 150
    JSR FUNCTION_DRAW_TILE
    STA $7EC65A
    ; Top left
    LDA !rhud_i_tl : STA !tile_info
    JSR FUNCTION_GET_BG3_BASE : CLC : ADC !rhud_v_tl : STA !vram_target
    LDA #$0000 : STA !affect_right_tile
    LDA #$00C8 : STA !healthCheck_Lower  ; 200
    LDA #$012C : STA !healthCheck_Upper  ; 300
    JSR FUNCTION_DRAW_TILE
    STA $7EC618
    ; Top right
    LDA !rhud_i_tr : STA !tile_info
    JSR FUNCTION_GET_BG3_BASE : CLC : ADC !rhud_v_tr : STA !vram_target
    LDA #$0001 : STA !affect_right_tile
    LDA #$00FA : STA !healthCheck_Lower  ; 250
    LDA #$015E : STA !healthCheck_Upper  ; 350
    JSR FUNCTION_DRAW_TILE
    STA $7EC61A
    RTS

; Return BG3 base VRAM address in the accumulator
FUNCTION_GET_BG3_BASE:
    LDA $005D : AND #$0F00 : ASL #4
    RTS

; Queues the tile to be used to the VRAM write table and returns the tile info. Arguments:
; - !tile_info (Function returns this if enough max reserves, otherwise empty tile)
; - !vram_target
; - !affect_right_tile
; - !healthCheck_Lower
; - !healthCheck_Upper
FUNCTION_DRAW_TILE:
    ; First check if the max reserves even reach this amount
    LDA !samus_max_reserves
    CMP !healthCheck_Lower       ; Example: If (100 - 0 > 0) { DRAW! }
    BEQ FDT_RETURN_EMPTY_TILE
    BPL FDT_UPPER_BAR_CHECK
FDT_RETURN_EMPTY_TILE:
    LDA #$2C0F
    RTS
FDT_UPPER_BAR_CHECK:
    STZ !affect_upper_bar
    LDA !healthCheck_Upper
    CMP !samus_reserves
    BPL FDT_SPECIAL_TILE_CHECK_LOWER
    LDA #$0001 : STA !affect_upper_bar
FDT_SPECIAL_TILE_CHECK_LOWER:
    ; if reserves are between (lower and lower + 50) or (upper and upper + 50) { special tile }
    LDA !healthCheck_Lower : CLC : ADC #$0032 : STA !special_helper ; !special_helper = Lower+50
    LDA !samus_reserves
    CMP !healthCheck_Lower
    BEQ FDT_SPECIAL_TILE_CHECK_UPPER : BMI FDT_SPECIAL_TILE_CHECK_UPPER ; if (reserves <= 0) { Continue } else { Check upper limit }
    CMP !special_helper
    BPL FDT_SPECIAL_TILE_CHECK_UPPER ; Continue
    BMI FDT_RETURN_SPECIAL_TILE
FDT_SPECIAL_TILE_CHECK_UPPER:
    LDA !healthCheck_Upper : CLC : ADC #$0032 : STA !special_helper
    LDA !samus_reserves
    CMP !healthCheck_Upper
    BEQ FDT_PREPARE_Y : BMI FDT_PREPARE_Y ; Continue
    CMP !special_helper
    BPL FDT_PREPARE_Y ; Continue
    ;BMI FDT_RETURN_SPECIAL_TILE
FDT_RETURN_SPECIAL_TILE:
    JMP FUNCTION_CREATE_SPECIAL_TILE ; This function ends with RTS, so below is not continued
FDT_PREPARE_Y:
    ; Store current tile offset in Y
    LDY #TABLE_NEW_TILES
    LDA !affect_right_tile
    BEQ FDT_CALC_START
    TYA : CLC : ADC #$0010 : TAY ; If right column
FDT_CALC_START:
    LDA !healthCheck_Upper
    CMP !samus_max_reserves
    BPL FDT_HAS_HEALTH_FIRST_RESERVE
    TYA : CLC : ADC #$0040 : TAY ; If there are at least TWO reserves
    LDA !healthCheck_Upper
    CMP !samus_reserves
    BPL FDT_HAS_HEALTH_FIRST_RESERVE
    TYA : CLC : ADC #$0020 : TAY ; If the 2nd reserve has ANY health
FDT_HAS_HEALTH_FIRST_RESERVE:
    LDA !healthCheck_Lower
    CMP !samus_reserves
    BPL FDT_RETURN_TILE
    TYA : CLC : ADC #$0020 : TAY ; If the 1st reserve has ANY health
FDT_RETURN_TILE:
    LDX $0330
    LDA #$0010      : STA $00D0,x ; Number of bytes
    LDA #$8000      : STA $00D3,x ;\ (Write bank first)
    TYA             : STA $00D2,x ;} Source address
    LDA !vram_target : STA $00D5,x ; Destination in Vram
    TXA : CLC : ADC #$0007 : STA $0330 ; Update the stack pointer
    LDA !tile_info
    RTS

; Creates the sub-tile progress tile in VRAM
FUNCTION_CREATE_SPECIAL_TILE:
    ; Step 1: Copy the data of the tile that the special tile should be based on
    LDA #TABLE_NEW_TILES : STA !special_helper
FCST_DECIDE_RIGHT_TILE:
    LDA !affect_right_tile
    BEQ FCST_DECIDE_TWO_BARS
    LDA !special_helper : CLC : ADC #$0010 : STA !special_helper
FCST_DECIDE_TWO_BARS:
    LDA !healthCheck_Upper
    CMP !samus_max_reserves
    BPL FCST_MEMCPY
    LDA !special_helper : CLC : ADC #$0040 : STA !special_helper
FCST_DECIDE_FILL_LOWER_BAR:
    LDA !healthCheck_Upper
    CMP !samus_reserves
    BPL FCST_MEMCPY
    LDA !special_helper : CLC : ADC #$0020 : STA !special_helper
FCST_MEMCPY:
    PHB
    LDA #$000F             ; Copy 16 bytes
    LDX !special_helper     ; Source
    LDY #!special_tile_loc ; Destination
    MVN $807E
    PLB
FCST_PAINT_BAR:
    ; Step 2: Paint over the columns of the bar that should be filled
    ; First change data bank to 7E
    PHB : PEA $7E00 : PLB : PLB
    LDX !healthCheck_Upper
    LDY #!special_tile_loc
FCST_PAINT_BAR_DECIDE_OFFSET:
    ; When painting top bar, the first 8 bytes are affected
    ; When painting bottom bar, the last 8 bytes are affected
    LDA !affect_upper_bar
    BNE FCST_PAINT_COLUMNS
    INY #8
    LDX !healthCheck_Lower
FCST_PAINT_COLUMNS:
    ; X has health test
    ; Y has address
    LDA !affect_right_tile
    BNE FCST_PAINT_COLUMN_0
    JMP FCST_PAINT_COLUMN_2_SKIP_CHECK
FCST_PAINT_COLUMN_0:
    INX #4 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_1
    LDA $0002,y : AND #$7F7F : ORA #$0080 : STA $0002,y
    LDA $0004,y : AND #$7F7F : ORA #$0080 : STA $0004,y
FCST_PAINT_COLUMN_1:
    INX #9 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_2
    LDA $0002,y : AND #$BFBF : ORA #$0040 : STA $0002,y
    LDA $0004,y : AND #$BFBF : ORA #$0040 : STA $0004,y
FCST_PAINT_COLUMN_2:
    INX #9 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_3
FCST_PAINT_COLUMN_2_SKIP_CHECK:
    LDA $0002,y : AND #$DFDF : ORA #$0020 : STA $0002,y
    LDA $0004,y : AND #$DFDF : ORA #$0020 : STA $0004,y
FCST_PAINT_COLUMN_3:
    INX #9 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_4
    LDA $0002,y : AND #$EFEF : ORA #$0010 : STA $0002,y
    LDA $0004,y : AND #$EFEF : ORA #$0010 : STA $0004,y
FCST_PAINT_COLUMN_4:
    INX #9 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_5
    LDA $0002,y : AND #$F7F7 : ORA #$0008 : STA $0002,y
    LDA $0004,y : AND #$F7F7 : ORA #$0008 : STA $0004,y
    LDA !affect_right_tile : BEQ FCST_PAINT_COLUMN_5 ; Right side tiles stop here
    JMP FCST_DMA_SPECIAL_TILE
FCST_PAINT_COLUMN_5:
    INX #9 : CPX !samus_reserves : BPL FCST_PAINT_COLUMN_6
    LDA $0002,y : AND #$FBFB : ORA #$0004 : STA $0002,y
    LDA $0004,y : AND #$FBFB : ORA #$0004 : STA $0004,y
FCST_PAINT_COLUMN_6:
    INX #9 : CPX !samus_reserves : BPL FCST_DMA_SPECIAL_TILE
    LDA $0002,y : AND #$FDFD : ORA #$0002 : STA $0002,y
    LDA $0004,y : AND #$FDFD : ORA #$0002 : STA $0004,y
FCST_PAINT_COLUMN_7:
    INX #9 : CPX !samus_reserves : BPL FCST_DMA_SPECIAL_TILE
    LDA $0002,y : AND #$FEFE : ORA #$0001 : STA $0002,y
    LDA $0004,y : AND #$FEFE : ORA #$0001 : STA $0004,y
FCST_DMA_SPECIAL_TILE:
    ; Step 3: Get the data over to the VRAM
    LDX $0330
    LDA #$0010             : STA $00D0,x ; Number of bytes
    LDA #$7E00             : STA $00D3,x ;\
    LDA #!special_tile_loc : STA $00D2,x ;}Source address
    LDA !vram_target        : STA $00D5,x ; Destination in Vram
    TXA : CLC : ADC #$0007 : STA $0330 ; Update the stack pointer
    PLB ; Restore data bank
    LDA !tile_info
    RTS

; Hook: HUD init
HOOK_HUD_INIT:
    STZ $0A06 ; Original code: Samus previous health = 0
    LDA #$FFFF : STA !samus_previous_reserves
    RTS

; Hook: Door transition
HOOK_DOOR_TRANSITION:
    STA $2100 ; Original code
    REP #$30
    LDA #$FFFF : STA !samus_previous_reserves
    JMP FUNCTION_DRAW_RESERVE_HUD

; REPAINTS: Rewrite tiles in VRAM immediately after tileset is reloaded
FUNCTION_REPAINT:
    LDA #$FFFF : STA !samus_previous_reserves
    JSR FUNCTION_DRAW_RESERVE_HUD
    RTL

org $809668
    JMP HOOK_DOOR_TRANSITION

org $828D4B ; Pause
    JSR FUNCTION_PAUSE_REPAINT_HELPER

org $82939C ; Unpause
    JSR FUNCTION_PAUSE_REPAINT_HELPER

org $82FF00
FUNCTION_PAUSE_REPAINT_HELPER:
    INC $0998
    JSL FUNCTION_REPAINT
    RTS

org !bank_82_free_space_start
FUNCTION_KRAID_LEAVE_REPAINT_BG3:
    PHA
    PHP
    JSR FUNCTION_KRAID_REPAINT
    PLP
    PLA
    STA $5A ; Original code
    STA $5B
    RTL

FUNCTION_KRAID_ENTER_REPAINT_BG3:
    LDA #$8000
    CPY #$B817                  ; Kraid (alive) initial cmd 0008?
    BEQ .hook
    CPY #$B842                  ; Kraid (dead) initial cmd 0008?
    BEQ .hook
    JMP $E606                   ; if not, return to hook point
    
.hook    
    TSB $05BC
.spin
    BIT $05BC
    BMI .spin                   ; process hooked VRAM update
    PHY                         ; save cmd ptr
    PHP
    SEP #$20
    LDA #$02
    STA $5E                     ; update BG3 base address now
    JSR FUNCTION_KRAID_REPAINT
    PLP
    PLY                         ; restore cmd ptr
    JMP $E60E                   ; return to end of hooked func

FUNCTION_KRAID_REPAINT:
    PHB
    REP #$30
    PEA $8000 : PLB : PLB
    JSL FUNCTION_REPAINT
    PLB
    RTS

warnpc !bank_82_free_space_end

; Hook: On reserve pickup
org $848986
    JSR HOOK_RESERVE_PICKUP

; Hook: Prevents blinking of reserve HUD on BG3 base address update (exiting Kraid)
org $8883DC
    JSL FUNCTION_KRAID_LEAVE_REPAINT_BG3

; Hook: Prevents blinking of reserve HUD on BG3 base address update (entering Kraid)
org $82E603
    JMP FUNCTION_KRAID_ENTER_REPAINT_BG3

org !bank_84_free_space_start
HOOK_RESERVE_PICKUP:
    LDA #$FFFF : STA !samus_previous_reserves
    LDA $09D4 ; Original code
    RTS
warnpc !bank_84_free_space_end

; When enterng and leaving Kraid's room, the original tiles are drawn for a few frames, so make them empty
org $E2C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E2C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E2C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

org $E3C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E3C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E3C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

org $E4C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E4C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E4C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

org $E5C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E5C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E5C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

org $E6C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E6C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E6C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00

org $E7C330
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E7C460
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
org $E7C4C0
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
    db $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
