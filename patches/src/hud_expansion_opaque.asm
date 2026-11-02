lorom

!bank_80_free_space_start = $80E540
!bank_80_free_space_end = $80E580
!bank_85_free_space_start = $85A8B0
!bank_85_free_space_end = $85A9B0

; Set this to the relative path between the assembler and this file (eg. ROMProject/ASM)
; If the assembler is in the same directory, then leave it as '.'
!this_dir = HUD_Expansion

; This patch clears up tiles for additional map and hud graphics.
; In addition, the top row of the HUD is now editable.
;
; This is largely accomplished by doing 3 things:
;   1) The file select graphics are moved to a new location.
;      It can be changed to any free space at `FileSelect_BG1_GFX`
;   2) The FX graphics and message box letters are dynamically loaded
;      the bottom row of the HUD graphics. If more FX graphics types
;      are desired, the loading routines may need to be updated here.
;      Note: The bottom row of the HUD graphics is reserved. Do not 
;      use these tiles for the hud as they may be either the FX or the
;      message box letters.
;   3) The PB/CF HDMA are made more compact to allow more space for the
;      top row of the hud.
;
; Editing map graphics:
; - Graphics with an `O` are free on both the HUD and pause screen and
;   can thus be used as map tiles.
; - Graphics with an `X` can can be used but not for the map tiles.
; - The graphic tile with an '*' can only be used for message boxes.
; - Ensure that all map tiles line up on the HUD and Pause graphics
;
; Advanced map graphics editing:
; - Graphic tiles from 0x00-0xDF can all be used for map tiles if
;   they are free on both the HUD and Pause. More tiles can be made
;   available if some HUD elements are removed such as the reserve
;   AUTO and "energy" text tiles. The "%" and "x" tiles on the top
;   row are not used in vanilla and are safe to remove. 
; - Graphic tiles from 0xE0-0xFF are dynamically loaded for FX/Msg.
;   These can be completely replaced with different message box
;   graphics as long as message.ttb is updated appropriately.
; - HUD graphics 0x100-0x1FF is not normally loaded into VRAM. This
;   patch uses this space for the HUD dynamically loaded graphics.
;   If more FX graphics are added, this region is a perfect place to
;   add them.
; (Opaque Version)
; - If the text is not intended to be used on the HUD in any way and 
;   the extended message box tiles are not needed, then it should be 
;   possible to move the text tiles down over the extended tiles and 
;   move the associated pause screen tiles to a free location. The 
;   letters will only be loaded during message boxes, clearing up 32 
;   more tiles. This would require changing the many of the TTB 
;   included in this patch, including message.ttb and the pause screen 
;   ttb.
; (Transparent Version)
; - If the text is not intended to be used on the HUD in any way then
;   it should be safe to overwrite them first rows of opaque letters.
;   However to use them for map tiles, the tiles in the in the map
;   screen graphics will need to be moved elsewhere. Make sure to update
;   the area names appropriately.
;
; Free space used:
;   - Load_FX: Bank 89
;   - FileSelect_BG1_GFX: Any bank
;
; Since this patch includes many TTB and GFX data, if SMART is being
; used then the appropriate data should be added to the export directory
; and removed from this patch. Also be aware if any other patches used
; that edit any of the included TTB/GFX data.
;
; Known bugs
;  - The elevator map tile and left/right arrow are moved, so they will 
;    be the wrong tile in vanilla maps.
;  - The samus figure in the message box that appears in the Bomb item
;    message box is removed. (opaque version)
;
; This asm was written by TestRunner based on CleanHUD by BlackFalcon,
; which in turn based on the DC Map patch. The main changes in this 
; version is a couple more free tiles, almost all the extended message
; box graphics is available, and the patch is entirely in ASM for
; ease of editting.

;----------------------------
; Load GFX/TTB Data
;----------------------------

; SMART already exports this data, so if SMART is being used then
; it is recommended to copy these files into "ROMProject\Export\Maps\"
; and remove these 2 incbins and map name tiles
org $B68000
    incbin !this_dir/Data/pausemap.gfx
org $9AB200
;    incbin !this_dir/Data/Opaque/HUD.gfx
    incbin !this_dir/Data/Opaque/HUD_vanilla.gfx   ; restores vanilla font for message boxes

org $82966F  ; Area Map Names
    DW $300F, $300F, $30C2, $30D1, $30C0, $30D3, $30C4, $30D1, $30C8, $30C0, $300F, $300F ; Crateria
    DW $300F, $300F, $30C1, $30D1, $30C8, $30CD, $30D2, $30D3, $30C0, $30D1, $300F, $300F ; Brinstar
    DW $300F, $300F, $30CD, $30CE, $30D1, $30C5, $30C0, $30C8, $30D1, $300F, $300F, $300F ; Norfair
    DW $30D6, $30D1, $30C4, $30C2, $30CA, $30C4, $30C3, $300F, $30D2, $30C7, $30C8, $30CF ; Wrecked Ship
    DW $300F, $300F, $30CC, $30C0, $30D1, $30C8, $30C3, $30C8, $30C0, $300F, $300F, $300F ; Maridia
    DW $300F, $300F, $30D3, $30CE, $30D4, $30D1, $30C8, $30C0, $30CD, $300F, $300F, $300F ; Tourian / debug
    DW $300F, $300F, $300F, $30C2, $30CE, $30CB, $30CE, $30CD, $30D8, $300F, $300F, $300F ; Ceres

org $899800
    incbin !this_dir/Data/pb_hdma.bin

org $81B14B
    incbin !this_dir/Data/map_load_hud.ttb

org $82D521
    incbin !this_dir/Data/samus_wireframe.ttb

org $85877F
    incbin !this_dir/Data/Opaque/messages.ttb

; FX tilemap
org $8A8000
    incbin !this_dir/Data/FX_lava.ttb
    incbin !this_dir/Data/FX_acid.ttb
    incbin !this_dir/Data/FX_water.ttb
    incbin !this_dir/Data/FX_spores.ttb
    incbin !this_dir/Data/FX_rain.ttb
    incbin !this_dir/Data/FX_fog.ttb

org $8E8000
    incbin !this_dir/Data/file_select_BG2.GFX

org $B6E000
    incbin !this_dir/Data/pause_BG2.ttb
    incbin !this_dir/Data/equipment.ttb

; free space
org $B8E000
FileSelect_BG1_GFX:
    incbin !this_dir/Data/file_select_BG1.GFX

; Equipment screen item tilemaps
org $82BF06
; MODE[MANUAL]
   DW $2519, $251A, $251B, $3D46, $3D47, $3D48, $3D49 
; RESERVE TANK
   DW $3EC8, $3EC9, $3ECA, $3ECB, $3ECC, $3ECD, $3ECE
; [MANUAL]
   DW $3D46, $3D47, $3D48, $3D49 
; [ AUTO ]
   DW $3D56, $3D57, $3D58, $3D59
; oCHARGE
   DW $0AFF, $0AD8, $0AD9, $0ADA, $0AE7 
; oICE
   DW $0AFF, $0ADB, $0ADC, $0AD4, $0AD4 
; oWAVE
   DW $0AFF, $0ADD, $0ADE, $0ADF, $0AD4 
; oSPAZER
   DW $0AFF, $0AE8, $0AE9, $0AEA, $0AEB 
; oPLASMA
   DW $0AFF, $0AEC, $0AED, $0AEE, $0AEF
; oVARIA SUIT
   DW $0AFF, $0900, $0901, $0902, $0903, $0904, $0905, $0AD4, $0AD4 
; oGRAVITY SUIT
   DW $0AFF, $0AD0, $0AD1, $0AD2, $0AD3, $0903, $0904, $0905, $0AD4 
; oMORPHING BALL
   DW $0AFF, $0920, $0921, $0922, $0923, $0917, $0918, $090F, $091F 
; oBOMBS
   DW $0AFF, $0AD5, $0AD6, $0AD7, $0AD4, $0AD4, $0AD4, $0AD4, $0AD4 
; oSPRING BALL
   DW $0AFF, $0910, $0911, $0912, $0913, $0914, $0915, $0916, $0AD4
; Unused
   DW $0000
; oSCREW ATTACK
   DW $0AFF, $0AE0, $0AE1, $0AE2, $0AE3, $0AE4, $0AE5, $0AE6, $0AD4 
; oHI-JUMP BOOTS
   DW $0AFF, $0930, $0931, $0932, $0933, $0934, $0935, $0936, $0AD4 
; oSPACE JUMP
   DW $0AFF, $0AF0, $0AF1, $0AF2, $0AF3, $0AF4, $0AF5, $0AD4, $0AD4 
; oSPEED BOOSTER
   DW $0AFF, $0924, $0925, $0926, $0927, $0928, $0929, $092A, $092B 
; oHYPER
   DW $0AFF, $0937, $0938, $0939, $092F, $0AD4, $0AD4, $0AD4, $0AD4
; Blank placeholder
   DW $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F


;------------------------------------
; Allow top row of HUD to be edited
;------------------------------------

org $809632
; allow doing the door transition DMA without forcing blanking of layers
ExecuteTransitionDMAWithBlank:
    XBA
    SEP #$20
    CLV
    STA $2100 ; sets the screen to be blacked. A = 0x80
ExecuteTransitionDMAWithoutBlank:

org $809661
    ; skip reverting screen visibility if wasn't blacked
    STZ $05BD
    BVS $07

; previously loaded to check if transition DMA should happen
; but since the value (0x80) matches the screen blank value,
; we pass this into the JSR to save a few bytes of code
org $80978E
    LDA $05BC ; load to A instead of X
org $80980A
    LDA $05BC ; load to A instead of X

; load top row in init
org $809AA3
    LDA $988B,X ; 98CB
    STA $7EC5C8,X ; 7EC608
    INX
    INX
    CPX #$0100   ; 00C0

; add top hud row to d table transfer
org $809CAD
    LDA #$0100 ; #$00C0
    STA $D0,x
    INX
    INX
    LDA #$C5C8  ; C608
org $809CC1
    LDA #$5800 ; 5820

org $80A0ED
    ; add an injection to run the transition DMA at gameplay load
    JSR TransitionDMA_CalculateLayer2XPos ; A2F9

org !bank_80_free_space_start
; Added function
TransitionDMA_CalculateLayer2XPos:
    ; New entry point into 'Calculate layer 2 X position'
    ; Executes the transition DMA prior to the normal call
    PHP
    LDX $05BC
    BPL $05
    SEP #$60
    JSR ExecuteTransitionDMAWithoutBlank
    PLP
    JSR $A2F9  ; run hi-jacked instruction
    RTS
assert pc() <= !bank_80_free_space_end
;warnpc $80A2F9 : padbyte $FF : pad $80A2F9


;---------------------------------------
; Dynamically load FX/Message Graphics
;---------------------------------------

org $858090
    JSR ClearLayer3ForMessageOpen ; 81F3 Clear message box BG3 tilemap
org $8580AA
    JSR ClearLayer3ForMessageClose ; 81F3 Clear message box BG3 tilemap
org $8580E8
    JSR ClearLayer3ForMessageClose ; 81F3 Clear message box BG3 tilemap

org !bank_85_free_space_start
ClearLayer3ForMessageOpen:
    PHP
    REP #$30
    PHY
    LDA #$4600 : STA $2116
    LDA #$BE00
    LDX #$0400
    LDY #$009A
    JSR ClearLayer3_Go
    BRA ClearLayer3_Finish
ClearLayer3ForMessageClose:
    PHP
    REP #$30
    PHY
    LDX #$4700 : STX $2116
    LDA $196E            ; FX type
    BEQ ClearLayer3_Finish
    CMP #$0026
    BEQ ClearLayer3_Water
    CMP #$000C
    BEQ ClearLayer3_Fog
    BCS ClearLayer3_Finish
    CMP #$0006
    BEQ ClearLayer3_Water
    LDX $1F17
    LDA $86FFFE,X
    LDX $1F3B
    LDY #$0087
    BRA ClearLayer3_CloseGo
ClearLayer3_Fog:
    LDA #$C400
    BRA ClearLayer3_FX
ClearLayer3_Water:
    LDA #$C200
ClearLayer3_FX:
    LDX #$0200
    LDY #$009A
ClearLayer3_CloseGo:
    JSR ClearLayer3_Go
ClearLayer3_Finish:
    PLY
    PLP
    JSR $81F3            ; displaced code
    RTS
ClearLayer3_Go:
    STA $4302
    STX $4305
    LDX #$1801 : STX $4300
    SEP #$20
    TYA
    STA $4304
    LDA #$80 : STA $2115
    LDA #$01 : STA $420B
    REP #$20
    RTS
assert pc() <= !bank_85_free_space_end

; load FX
org $89AC31
    JSR Load_FX ; LDA $ABF0,y FX tilemap pointer = [$ABF0 + [FX type]]

; free space in Bank 89
org $89AF00 ; A = FX type
Load_FX:
    CMP #$0026 : BEQ LoadFX_Water ; type = golden statues
    CMP #$0006 : BEQ LoadFX_Water ; type = water
    CMP #$000C : BNE LoadFX_Return ; not fog -> nothing to load
    LDA #$9AC4 ; fog
    BRA LoadFX_Continue
LoadFX_Water:
    LDA #$9AC2
LoadFX_Continue:
    STA $05C1 ; dma ram address
    LDA #$0047
    STA $05BF ; dma vram address
    LDA #$0200
    STA $05C3 ; dma size
    LDA #$0080
    STA $05BD ; queue transition dma
LoadFX_Return:
    LDA $ABF0,y ; Displaced code
    RTS
assert pc() <= $89AF60

;----------------------------
; Repoint Data
;----------------------------

; move titlescreen gfx data
org $818E34
    DL FileSelect_BG1_GFX ;B6C000

; Change Graphics DMA size to not get the bottom row of 1 screen
org $828ED1
    DW $0E00   ; 2000

; animated tile object for FX
; modify the VRAM destination to match the rearranged tilemap (bottom row)
org $8782AF ; lava
    DW $4700 ; 4280
org $8782CD ; acid
    DW $4700 ; 4280
org $8782EB ; rain
    DW $4700 ; 4280
org $878301 ; spores
    DW $4700 ; 4280

; The PB/CF HDMA tables are shortened in order to make more space in WRAM
; for the added row of the HUD tilemap.
org $888B18 ; pb explosion 
    STA $7EC5C6 ; 7EC606
    LDA #$00
    STA $7EC5C7 ; 7EC607
org $88A2E8 ; crystal flash
    STA $7EC5C6 ; 7EC606
    LDA #$00
    STA $7EC5C7 ; 7EC607

;----------------------------
; Rearranged tiles
;----------------------------

; 'blank' tile when clearing layer 3 during pause
org $80A214
    LDA #$180F ; 184E

; 'blank' tile when clearing layer 3
org $81A7AE
    LDA #$280F ; 2801

; 'blank' tile for load map
org $829580
    LDA #$000E ; 000F

; 'blank' tile for clearing layer 3 (BG Data Instruction)
org $82E569
    LDA #$180F ; 184E
    
org $809DBF
; Health
    DW $2C00, $2C01, $2C02, $2C03, $2C04, $2C05, $2C06, $2C07, $2C08, $2C09
; Ammo
    DW $2C00, $2C01, $2C02, $2C03, $2C04, $2C05, $2C06, $2C07, $2C08, $2C09

org $858000
    ; large message box border
    DW $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F
    DW $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $000E, $000E, $000E
    ; small message box border
    DW $000E, $000E, $000E, $000E, $000E, $000E, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F
    DW $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $2C0F, $000E, $000E, $000E, $000E, $000E, $000E, $000E

; Digit tilemap for reserves because the numbers are moved
org $828FB6
    ADC #$0800 ; 0804
org $828FC1
    ADC #$0800 ; 0804
org $828FCB
    ADC #$0800 ; 0804
org $82B3BB
    ADC #$0800 ; 0804
org $82B3C6
    ADC #$0800 ; 0804
org $82B3D0
    ADC #$0800 ; 0804