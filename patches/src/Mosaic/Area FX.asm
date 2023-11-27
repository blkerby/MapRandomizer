lorom

; Use tileset index to figure out glows instead of area
org $89AC62
  JSR GetTilesetIndex ;LDA $079F
  ASL
  TAY
  LDA.w GlowTypeTable,Y ;DB is $83

org $89AC98
  JSR GetTilesetIndex ;LDA $079F
  ASL
  TAY
  LDA.w AnimTypeTable,Y

; Force on excape glow bits during escape event
org $89AB90
  JMP ForceGlowMask
org $89ABA3
  JMP ForceGlowMask
org $89AC57
  JSR MaskGlowBits
  ;LDA $000D,X
  ;AND #$00FF

; Turn off rain if pbs have been collected
org $89AC25
  JSR GetFxType
  ;LDA $0009,X
org $89ABFB
  JSR GetFxPaletteBlend
  ;AND #$00FF

; use surface new to set max height for lightning with rain fx
org $8DEC59
  LDA $0AFA
  CMP $197A
  ;LDA $0AFA
  ;CMP #$0380

org $89AF60 ;free space
GetTilesetIndex:
  PHX
  LDX $07BB
  LDA $8F0003,X
  PLX
  AND #$00FF
  RTS

CallGlowHandler:
  PHB
  PHK
  PLB
  JSR GetTilesetIndex
  PHX
  ASL
  TAX
  JSR (GlowHandlerTable,X)
  PLX
  PLB
  RTS

MaskGlowBits:
  LDA $000D,X
  AND #$00FF
  STA $196A
  JSR CallGlowHandler
  RTS

ForceGlowMask:
  LDA #$0000
  STA $196A
  JSR CallGlowHandler
  AND #$00FF
  STA $196A
  BEQ ForceGlowMask_Exit

  JSR GetTilesetIndex
  ASL
  TAY
  LDA.w GlowTypeTable,Y ;DB is $83
  STA $AF
  LDY #$0000

ForceGlowMask_Loop:
  LSR $196A
  BCC ForceGlowMask_Next
  LDA ($AF),Y
  PHY
  TAY
  JSL $8DC4E9
  PLY
ForceGlowMask_Next:
  INY
  INY
  CPY #$0010
  BNE ForceGlowMask_Loop

ForceGlowMask_Exit:
  PLB
  PLP
  RTL

GlowHandlerTable:
  DW Handler_Area_0a, Handler_Area_0a ;Crateria Surface
  DW Handler_Area_0b, Handler_Area_0b ;Inner Crateria
  DW Handler_Area_3, Handler_Area_3 ;Wrecked Ship
  DW Handler_Area_1, Handler_Area_1 ;Brinstar
  DW Handler_Area_1 ;Tourian Statues Access
  DW Handler_Area_2, Handler_Area_2 ;Norfair
  DW Handler_Area_4, Handler_Area_4 ;Maridia
  DW Handler_Area_5, Handler_Area_5 ;Tourian
  DW Handler_Area_6, Handler_Area_6, Handler_Area_6, Handler_Area_6, Handler_Area_6, Handler_Area_6 ;Ceres
  DW Handler_Area_6, Handler_Area_6, Handler_Area_6, Handler_Area_6, Handler_Area_6 ;Utility Rooms
  ;Bosses
  DW Handler_Area_1 ;Kraid
  DW Handler_Area_2 ;Crocomire
  DW Handler_Area_4 ;Draygon
  DW Handler_Area_1 ;SpoSpo
  DW Handler_Area_3 ;Phantoon

ProcessEscapeMask:
  LDA #$000E
  JSL $808233
  BCC ProcessEscapeMask_Default
  LDA EscapeGlowTable,Y
  BIT $196A ; If the excape glows are already handled by the fx, use that.
  BNE ProcessEscapeMask_Default
  ORA $196A
  AND EscapeMaskTable,Y ; Turn off the basic tileset specific glow in tilesets with escape glows.
  RTS
ProcessEscapeMask_Default:
  LDA $196A
  RTS

EscapeGlowTable:
  DB $06 ;Crateria Surface
  DB $18 ;Inner Crateria
  DB $1D ;Tourian

EscapeMaskTable:
  DB $FE ;Crateria Surface
  DB $FF ;Inner Crateria
  DB $FD ;Tourian

Handler_Area_0a:
  LDY #$0000
  JSR ProcessEscapeMask

  LDY $09D0
  BEQ +
  AND #$00FE
  RTS
+
  RTS

Handler_Area_0b:
  LDY #$0001
  JSR ProcessEscapeMask
  RTS

Handler_Area_3:
  LDA #$0058
  JSL $808233
  BCC +
  LDA $196A
  ORA #$0001
  RTS
+
  LDA $196A
  AND #$00FE
  RTS

Handler_Area_5:
  LDY #$0003
  JSR ProcessEscapeMask
  RTS

Handler_Area_1:
Handler_Area_2:
Handler_Area_4:
Handler_Area_6:
  LDA $196A
  RTS

GetFxType:
  LDA $0009,X
  AND #$00FF
  CMP #$000A
  BEQ GetFxType_Rain
  CMP #$000C
  BEQ GetFxType_Fog
GetFxType_Default:
  LDA $0009,X
  RTS
GetFxType_Fog:
  LDA #$0000
  JSL $808233
  BCC GetFxType_Default
  BRA GetFxType_Remove
GetFxType_Rain:
  LDA $09D0
  BEQ GetFxType_Default
GetFxType_Remove:
  LDA #$0000
  RTS

GetFxPaletteBlend:
  AND #$00FF
  PHA
  LDA $0009,X
  AND #$00FF
  CMP #$000A
  BEQ GetFxPaletteBlend_Rain
  CMP #$000C
  BEQ GetFxPaletteBlend_Fog
GetFxPaletteBlend_Default:
  PLA
  RTS
GetFxPaletteBlend_Fog:
  LDA #$0000
  JSL $808233
  BCC GetFxPaletteBlend_Default
  BRA GetFxPaletteBlend_Remove
GetFxPaletteBlend_Rain:
  LDA $09D0
  BEQ GetFxPaletteBlend_Default
GetFxPaletteBlend_Remove:
  PLA
  LDA #$0000
  RTS

; swap some tileset indexes based on asleep/off
org $82DEFD
  JSL CheckTileset
  ;AND #$00FF
  ;ASL
org $8AB500 ;free space (due to scrolling sky asm)
CheckTileset:
  AND #$00FF
  CMP #$0002
  BEQ InnerCrateriaAwake
  CMP #$0003
  BEQ InnerCrateriaAwake
  CMP #$0004
  BEQ WreakedShipAwake
  CMP #$0005
  BEQ WreakedShipAwake
  ASL
  RTL
InnerCrateriaAwake:
  LDA #$0000
  JSL $808233
  BCS +

  LDA $079F ; area index
  XBA
  ORA $079D ; room index
  CMP #$0013 ;OLD TOURIAN BOSS ROOM
  BEQ +
  CMP #$012D ;MINI KRAID HALLWAY
  BEQ +
  CMP #$0411 ;PLASMA BEAM ROOM
  BEQ +
  CMP #$0247 ;NINJA PIRATES BOSS ROOM
  BEQ +

  LDA #$0006 ;zebes asleep (tileset 3)
  RTL
+
  LDA #$0004 ;zebes awake (tileset 2)
  RTL
WreakedShipAwake:
  LDA #$0058
  JSL $808233
  BCC +
  LDA #$0008 ;Phantoon defeated (tileset 4)
  RTL
+
  LDA #$000A ;Phantoon lurking (tileset 5)
  RTL

UpdateSandFloorColors: ;DB is 8D
  PHX
  PHY
  LDY #$0000
  LDX #$0048
-
  LDA $F4EF,Y
  STA $7EC200,X
  INX
  INX
  INY
  INY
  CPY #$0010
  BMI -
  PLY
  PLX
  RTL
UpdateHeavySandColors: ;DB is 8D
  PHX
  PHY
  LDY #$0000
  LDX #$0050
-
  LDA $F547,Y
  STA $7EC200,X
  INX
  INX
  INY
  INY
  CPY #$0008
  BMI -
  PLY
  PLX
  RTL

; Dynamically load tiles on room init
LoadSpeecialRoomTiles:
  PHK
  PLB
; TODO, make some kind of check here to load different data
LoadSpeecialRoomTiles_Tube:
  LDY $0330
  LDA #$03E0
  STA $00D0,Y
  LDA #$8A00
  STA $00D3,Y
  LDA.w #TubeGfx_1
  STA $00D2,Y
  LDA #$3E00 ; area after the CRE
  STA $00D5,Y
  TYA
  CLC
  ADC #$0007
  TAY
  LDA #$0400
  STA $00D0,Y
  LDA #$8A00
  STA $00D3,Y
  LDA.w #TubeGfx_2
  STA $00D2,Y
  LDA #$2400 ; overwrite vileplumes
  STA $00D5,Y
  TYA
  CLC
  ADC #$0007
  STA $0330

  LDA #$8A00
  STA $0605
  LDA #LoadSpeecialRoomTiles_UnpauseHook
  STA $0604

  RTL

LoadSpeecialRoomTiles_UnpauseHook:
  PHP
  JSL $80836F ; force blank and lag a frame
  JSL LoadSpeecialRoomTiles_Tube
  JSL $808C83 ; process dma queue immediatly
  JSL $808382 ; unforce blank and lag a frame
  PLP
  RTL


org $8AB700
TubeGfx_1:
  DB $3F, $FF, $9F, $FF, $E7, $FF, $79, $FF, $1E, $FF, $87, $FF, $E1, $FF, $F8, $FF, $FF, $01, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $7F, $FF, $9F, $FF, $E7, $FF, $79, $FF, $FF, $9F, $FF, $67, $FF, $19, $FF, $06, $FF, $01, $FF, $00, $FF, $00, $FF, $00
  DB $1F, $FF, $07, $FF, $C1, $3F, $30, $CF, $0E, $F1, $C1, $3E, $F0, $0F, $3E, $C1, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $1E, $FF, $C7, $FF, $F1, $FF, $3E, $FF, $07, $FF, $81, $7F, $70, $8F, $0C, $F3, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $07, $F8, $E0, $1F, $1E, $E1, $81, $7E, $F8, $07, $FF, $00, $FF, $00, $1F, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $C3, $3C, $F8, $07, $0E, $F1, $E1, $1E, $1C, $E3, $83, $7C, $F0, $0F, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $C0, $00, $3E, $00, $01, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $00, $FF, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $FF, $FF, $FF
  DB $FF, $00, $07, $00, $F0, $00, $0F, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $FF
  DB $FF, $FF, $FF, $FF, $FF, $FF, $F4, $E6, $FC, $F9, $B9, $BF, $E7, $FF, $9E, $FF, $FF, $79, $FF, $E6, $FF, $98, $FF, $60, $FF, $80, $FF, $00, $FF, $04, $FF, $18
  DB $F6, $FC, $CC, $FC, $9E, $DC, $2E, $3E, $FF, $7E, $EE, $F4, $84, $FC, $18, $F8, $FE, $84, $FE, $08, $FF, $18, $FF, $2C, $FF, $0C, $FE, $04, $FE, $04, $FC, $08
  DB $78, $FF, $E3, $FF, $97, $E6, $7E, $F8, $E8, $F7, $89, $FE, $0E, $F1, $F4, $CF, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00
  DB $FC, $EE, $86, $1C, $84, $FC, $0A, $F6, $77, $8E, $8E, $7F, $5E, $FC, $7C, $BC, $FE, $2C, $FE, $00, $FE, $00, $FF, $00, $FF, $02, $FF, $04, $FE, $48, $FE, $38
  DB $DB, $3C, $3F, $E0, $71, $8F, $86, $7B, $3C, $C4, $C5, $3E, $0B, $F0, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $03, $FF, $00, $FF, $04, $FF, $00, $FF, $00
  DB $EC, $18, $1C, $FC, $FC, $FE, $BE, $7F, $0F, $EE, $E7, $1E, $36, $46, $D7, $2E, $FC, $08, $FC, $08, $FE, $F8, $FF, $1C, $FF, $0C, $FF, $0C, $FF, $04, $FF, $04
  DB $FF, $80, $F8, $18, $0F, $07, $B2, $43, $0C, $0E, $38, $00, $C1, $04, $00, $DD, $FF, $00, $FF, $00, $FF, $03, $FF, $01, $FF, $06, $FF, $00, $FF, $00, $FF, $DD
  DB $06, $04, $7E, $0C, $8C, $8C, $4C, $4C, $38, $38, $18, $18, $BC, $58, $78, $B8, $FE, $04, $FE, $08, $FE, $08, $FE, $48, $FC, $30, $FC, $10, $FC, $50, $FC, $B0
  DB $00, $00, $00, $00, $00, $00, $00, $00, $03, $03, $0F, $0E, $3F, $3F, $FF, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $03, $01, $0F, $06, $3F, $17, $FF, $3E
  DB $30, $68, $18, $14, $1C, $3A, $7A, $FE, $FC, $FA, $FC, $F4, $DC, $FC, $FD, $F6, $78, $00, $1C, $04, $3E, $08, $FE, $7A, $FE, $B8, $FC, $E4, $FD, $98, $FF, $64
  DB $C0, $C0, $F0, $F0, $FC, $FC, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $C0, $C0, $F0, $F0, $FC, $FC, $FF, $7F, $FF, $9F, $FF, $67, $FF, $19, $FF, $06
  DB $00, $00, $00, $00, $00, $00, $00, $00, $C0, $C0, $F0, $F0, $FC, $FC, $FF, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $C0, $C0, $F0, $F0, $FC, $FC, $FF, $7F
  DB $88, $D4, $5E, $FD, $FF, $FB, $7F, $7F, $DF, $FF, $F7, $FF, $FF, $FF, $1E, $1E, $FC, $04, $FF, $08, $FF, $A0, $7F, $76, $FF, $89, $FF, $77, $FF, $FF, $1E, $1E
  DB $BC, $5E, $97, $EB, $DF, $FE, $FF, $FF, $FE, $F7, $FF, $FF, $7E, $7C, $18, $18, $FE, $00, $FF, $00, $FF, $08, $FF, $9F, $FF, $C9, $FF, $FF, $7E, $78, $18, $18
  DB $FF, $B0, $80, $0C, $B0, $FF, $FF, $30, $30, $30, $30, $30, $30, $30, $30, $FF, $FF, $B0, $FF, $0C, $FF, $B0, $FF, $30, $FF, $30, $FF, $30, $FF, $30, $FF, $FF
  DB $FB, $74, $5F, $30, $70, $F0, $FB, $04, $00, $00, $00, $00, $04, $00, $00, $FF, $FF, $70, $FF, $10, $FF, $70, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $FF
  DB $88, $FF, $88, $F8, $88, $88, $A8, $8F, $8F, $8F, $88, $88, $88, $88, $F9, $88, $F8, $77, $FF, $77, $FF, $07, $F8, $07, $F8, $07, $FF, $07, $FF, $07, $FF, $07
  DB $00, $FF, $30, $00, $40, $00, $00, $FF, $FF, $FF, $80, $00, $80, $00, $00, $00, $00, $FF, $F0, $F0, $C0, $C0, $00, $FF, $00, $FF, $80, $80, $80, $80, $00, $00
  DB $FF, $FF, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $FF, $FF, $FF, $00, $00, $00, $00, $00, $00
  DB $F0, $F0, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $0F, $FF, $00, $00, $00, $00, $00, $FF, $FF, $FF, $00, $00, $00, $00, $00, $00
  DB $00, $00, $00, $00, $00, $00, $F0, $F0, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $0F, $FF, $FF, $FF, $00, $00, $00, $00, $00, $00
  DB $00, $00, $00, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $FF, $FF, $FF, $FF, $00, $00, $00, $00, $00, $00
  DB $8F, $F8, $F8, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $00, $FF, $88, $FF, $A8, $FF, $FF, $FF, $88, $FF, $FF, $FF, $FF, $FF, $FF
  ;DB $F7, $08, $88, $FF, $FF, $FF, $FF, $FF, $FF, $F7, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $00, $FF, $00, $FF, $08, $FF, $FF, $FF, $08, $FF, $FF, $FF, $FF, $FF, $FF
TubeGfx_2:
  DB $FF, $FF, $FF, $FF, $FF, $FD, $FA, $F8, $FF, $FD, $FF, $FA, $FF, $FD, $8A, $FA, $FF, $FF, $FF, $F8, $FF, $FF, $FF, $8F, $FF, $FF, $FF, $AF, $FF, $8F, $FD, $07
  DB $FF, $FF, $FF, $FF, $FF, $55, $AA, $00, $FF, $55, $FD, $A8, $AA, $00, $AA, $AA, $FF, $FF, $FF, $00, $FF, $FF, $FF, $FF, $FF, $FF, $FD, $FD, $AA, $AA, $55, $FF
  DB $FF, $FF, $FF, $FF, $FF, $55, $AA, $00, $FA, $50, $55, $00, $AA, $00, $AA, $AA, $FF, $FF, $FF, $00, $FF, $FF, $FF, $FF, $FA, $FA, $55, $55, $AA, $AA, $55, $FF
  DB $FF, $FF, $FF, $FF, $FE, $54, $AA, $00, $AA, $00, $55, $00, $AA, $00, $AA, $AA, $FF, $FF, $FF, $00, $FE, $FE, $FF, $FF, $AA, $AA, $55, $55, $AA, $AA, $55, $FF
  DB $FF, $FF, $FF, $FF, $AA, $00, $AA, $00, $AA, $00, $55, $00, $AA, $00, $AA, $AA, $FF, $FF, $FF, $00, $AA, $AA, $FF, $FF, $AA, $AA, $55, $55, $AA, $AA, $55, $FF
  DB $FF, $FF, $FF, $FF, $AA, $00, $AA, $00, $AA, $00, $55, $00, $AA, $00, $55, $00, $FF, $FF, $FF, $00, $AA, $AA, $FF, $FF, $AA, $AA, $55, $55, $AA, $AA, $FF, $FF
  DB $FE, $F8, $F8, $F8, $FE, $F8, $AD, $F8, $88, $F8, $F8, $F8, $F8, $F8, $88, $F8, $FE, $8E, $FD, $8D, $FE, $8E, $FD, $25, $F8, $00, $FF, $8F, $F8, $88, $F8, $00
  DB $AA, $00, $00, $00, $00, $00, $45, $00, $00, $00, $00, $00, $00, $00, $00, $00, $AA, $AA, $45, $45, $2A, $2A, $55, $55, $00, $00, $FF, $FF, $00, $00, $00, $00
  DB $AA, $00, $00, $00, $00, $00, $45, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $AA, $AA, $45, $45, $2A, $2A, $55, $55, $00, $00, $FF, $FF, $00, $00, $00, $00
  DB $AA, $00, $00, $00, $00, $00, $45, $00, $00, $00, $00, $00, $00, $00, $00, $00, $AA, $AA, $45, $45, $2A, $2A, $55, $55, $00, $00, $00, $00, $00, $00, $00, $00
  DB $AA, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $AA, $AA, $41, $41, $28, $28, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00
  DB $8C, $F8, $AC, $F8, $F8, $8F, $8F, $F8, $8F, $F8, $F8, $8F, $F8, $8F, $AF, $F8, $FC, $04, $FC, $04, $F8, $07, $FF, $07, $FF, $07, $F8, $07, $F8, $07, $FF, $07
  DB $00, $00, $00, $00, $FF, $FF, $00, $00, $FF, $00, $3F, $FF, $00, $FF, $FF, $00, $00, $00, $00, $00, $00, $FF, $00, $00, $FF, $FF, $00, $FF, $00, $FF, $FF, $FF
  DB $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FC, $FC, $FF, $FF, $F0, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $03, $FF, $00, $FF, $F0, $F0
  DB $00, $00, $00, $00, $0F, $00, $00, $00, $00, $00, $00, $00, $C0, $C0, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $FF, $FF, $3F, $FF, $00, $00
  DB $00, $00, $00, $00, $C0, $00, $00, $00, $00, $00, $FF, $00, $03, $00, $00, $00, $00, $00, $00, $00, $C0, $C0, $00, $00, $00, $00, $FF, $FF, $FF, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00
  DB $8F, $F8, $FB, $88, $8A, $F8, $FF, $8F, $FA, $88, $D8, $AF, $F8, $8F, $89, $88, $FF, $07, $FF, $07, $FE, $06, $F8, $07, $FE, $06, $F8, $07, $F8, $07, $FF, $07
  DB $E0, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $3F, $FF, $0F, $FF, $00, $00, $E0, $E0, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $FF, $00, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $0F, $00, $00, $00, $C0, $C0, $F0, $F0, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $3F, $FF, $0F, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $C0, $00, $00, $00, $FC, $00, $0F, $00, $00, $00, $00, $00, $00, $00, $00, $00, $C0, $C0, $00, $00, $FC, $FC, $FF, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $0F, $0F, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $F0, $FF, $FF, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $F0, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $FF, $FF, $FF, $00, $00
  DB $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $F0, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $FF, $FF, $FF, $00, $00
  DB $F8, $88, $88, $88, $8F, $8F, $88, $8F, $A8, $88, $88, $8F, $8F, $F8, $88, $FF, $FF, $07, $FF, $07, $F8, $07, $F8, $07, $FF, $07, $F8, $07, $F8, $77, $FF, $77
  DB $80, $00, $60, $00, $FF, $FF, $00, $FF, $38, $00, $03, $FF, $FF, $00, $00, $FC, $80, $80, $E0, $E0, $00, $FF, $00, $FF, $F8, $F8, $00, $FF, $00, $FF, $FC, $FF
  DB $00, $00, $00, $00, $C0, $C0, $FF, $FF, $00, $00, $FC, $FC, $C0, $3F, $00, $00, $00, $00, $00, $00, $3F, $FF, $00, $FF, $00, $00, $03, $FF, $00, $FF, $00, $FF
  DB $00, $00, $00, $00, $07, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $00, $00, $00, $00, $FF, $FF, $FF, $FF, $00, $00, $FF, $FF, $00, $FF, $00, $FF
  DB $00, $00, $00, $00, $FC, $00, $00, $00, $00, $00, $0F, $00, $FF, $FF, $FF, $00, $00, $00, $00, $00, $FC, $FC, $FF, $FF, $00, $00, $FF, $FF, $00, $FF, $00, $FF
  DB $00, $00, $00, $00, $00, $00, $FF, $00, $00, $00, $FF, $00, $00, $00, $FF, $00, $00, $00, $00, $00, $00, $00, $FF, $FF, $00, $00, $FF, $FF, $FF, $FF, $00, $FF
  DB $00, $00, $00, $00, $FF, $FF, $F0, $FF, $FF, $00, $FF, $FF, $FF, $00, $00, $00, $00, $00, $00, $00, $00, $FF, $00, $FF, $FF, $FF, $00, $FF, $00, $FF, $00, $FF
  DB $F7, $08, $88, $FF, $FF, $FF, $FF, $FF, $FF, $F7, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $00, $FF, $00, $FF, $08, $FF, $FF, $FF, $08, $FF, $FF, $FF, $FF, $FF, $FF

org $8FC11B ; Room init code for ocean rooms no longer used due to scrolling sky
  JSL LoadSpeecialRoomTiles
  RTS

; force the sand glows to load their colors on init
org $8DF795
  DW #SandFloorColorsInit, $F4E9
org $8DF799
  DW #HeavySandColorsInit, $F541
org $8DC686 ; overwrite unused garbage data
SandFloorColorsInit:
  JSL UpdateSandFloorColors
  RTS
HeavySandColorsInit:
  JSL UpdateHeavySandColors
  RTS
warnpc $8DC696


; Move animation VRAM offsets
org $878279
  DW $2500 ; R_Tread
org $87827F
  DW $2500 ; L_Tread
org $878285
  DW $2400 ; Vileplum
org $87828B
  DW $26D0 ; SandHead
org $878291
  DW $26F0 ; SandFall

org $87C964
CeilingFrame1:
  DB $FA, $20, $2E, $C0, $9F, $00, $DD, $00, $6B, $07, $36, $1E, $90, $75, $E2, $C8, $00, $FF, $08, $FF, $1A, $FF, $58, $FF, $40, $FF, $02, $FC, $15, $E0, $2A, $C0
  DB $CF, $10, $73, $08, $6D, $02, $F7, $08, $43, $88, $81, $E8, $18, $5E, $25, $0F, $00, $FF, $20, $FF, $64, $FF, $74, $FF, $43, $FF, $00, $FF, $40, $1F, $A8, $07
CeilingFrame2:
  DB $FA, $20, $2E, $C0, $9F, $00, $DD, $00, $6B, $07, $36, $1C, $90, $60, $E2, $E2, $00, $FF, $08, $FF, $1A, $FF, $58, $FF, $40, $FF, $02, $FC, $15, $E0, $2A, $C0
  DB $CF, $10, $73, $08, $6D, $02, $F7, $08, $43, $88, $81, $E8, $18, $1E, $25, $A7, $00, $FF, $20, $FF, $64, $FF, $74, $FF, $43, $FF, $00, $FF, $40, $1F, $A8, $07
CeilingFrame3:
  DB $FA, $20, $2E, $C0, $9F, $00, $DD, $00, $6B, $07, $36, $1E, $90, $75, $E2, $C8, $00, $FF, $08, $FF, $1A, $FF, $58, $FF, $40, $FF, $02, $FC, $15, $E0, $2A, $C0
  DB $CF, $10, $73, $08, $6D, $02, $F7, $08, $43, $88, $81, $E8, $18, $5E, $25, $0F, $00, $FF, $20, $FF, $64, $FF, $74, $FF, $43, $FF, $00, $FF, $40, $1F, $A8, $07
CeilingFrame4:
  DB $FA, $20, $2E, $C0, $9F, $00, $DD, $00, $6B, $07, $36, $1C, $90, $60, $E2, $E2, $00, $FF, $08, $FF, $1A, $FF, $58, $FF, $40, $FF, $02, $FC, $15, $E0, $2A, $C0
  DB $CF, $10, $73, $08, $6D, $02, $F7, $08, $43, $88, $81, $E8, $18, $1E, $25, $A7, $00, $FF, $20, $FF, $64, $FF, $74, $FF, $43, $FF, $00, $FF, $40, $1F, $A8, $07

CeilingInstructionList:
  DW $000A, #CeilingFrame1
  DW $000A, #CeilingFrame2
  DW $000A, #CeilingFrame3
  DW $000A, #CeilingFrame4
  DW $80B7, #CeilingInstructionList

Ceiling_:
  DW #CeilingInstructionList, $0040, $26D0 ; Maridia sand ceiling

; Wait for Ws awake, not area boss awake
org $8781BA
  LDA #$0058
  JSL $808233

;Move vileplume tilemap tiles
org $849E0D
  DW $0002, $37D9, $87D8
  DB $FE, $00
  DW $0002, $83D8, $53D9
  DB $FE, $FF
  DW $0004, $23B8, $23B9, $27B9, $27B8
  DW $0000
org $849E45
  DW $0002, $07DB, $87DA
  DB $FE, $00
  DW $0002, $83DA, $03DB
  DB $FE, $FF
  DW $0004, $23BA, $23BB, $27BB, $27BA
  DW $0000
org $849E61
  DW $0002, $07DD, $87DC
  DB $FE, $00
  DW $0002, $83DC, $03DD
  DB $FE, $FF
  DW $0004, $23BC, $23BD, $27BD, $27BC
  DW $0000
org $849E7D
  DW $0002, $07DF, $87DE
  DB $FE, $00
  DW $0002, $83DE, $03DF
  DB $FE, $FF
  DW $0004, $23BE, $23BF, $27BF, $27BE
  DW $0000

org $849E99
  DW $0002, $3FD9, $8FD8
  DB $FE, $00
  DW $0002, $8BD8, $5BD9
  DB $FE, $01
  DW $0004, $2BB8, $2BB9, $2FB9, $2FB8
  DW $0000
org $849ED1
  DW $0002, $0FDB, $8FDA
  DB $FE, $00
  DW $0002, $8BDA, $0BDB
  DB $FE, $01
  DW $0004, $2BBA, $2BBB, $2FBB, $2FBA
  DW $0000
org $849EED
  DW $0002, $0FDD, $8FDC
  DB $FE, $00
  DW $0002, $8BDC, $0BDD
  DB $FE, $01
  DW $0004, $2BBC, $2BBD, $2FBD, $2FBC
  DW $0000
org $849F09
  DW $0002, $0FDF, $8FDE
  DB $FE, $00
  DW $0002, $8BDE, $0BDF
  DB $FE, $01
  DW $0004, $2BBE, $2BBF, $2FBF, $2FBE
  DW $0000

;Move Maridia tube tilemap tiles
org $8498D1
  DW $0001, $C7ED
  DW $0000
org $8498D7
  DW $0001, $87ED
  DW $0000
org $8498DD
  DW $0001, $83F2
  DW $0000
org $8498E3
  DW $000C, $83F2, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $87F2
  DB $00, $01
  DW $000C, $03EE, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $07EE
  DB $00, $02
  DW $000C, $03EF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $07EF
  DB $00, $03
  DW $000C, $0BEF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0FEF
  DW $0000
org $849953
  DW $0001, $03E2
  DB $00, $04
  DW $000C, $0BEE, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0FEE
  DB $00, $05
  DW $000C, $83F0, $83F1, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $87F1, $87F0
  DW $0000
org $849991
  DW $000C, $83F2, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $87F2
  DB $00, $01
  DW $000C, $03EE, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $07EE
  DB $00, $02
  DW $000C, $03EF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $07EF
  DW $0000
org $8499E5
  DW $0001, $03E2
  DB $00, $03
  DW $000C, $0BEF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0FEF
  DB $00, $04
  DW $000C, $0BEE, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0FEE
  DB $00, $05
  DW $000C, $83F0, $83F1, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $87F1, $87F0
  DW $0000


org $83B800

!NullGlow = $F745
;!Statue_3 = $F749
;!Statue_1 = $F74D
;!Statue_4 = $F751
;!Statue_2 = $F755
;!BT_Glow_ = $F759
;!GT_Glow_ = $F75D
!SamusHot = $F761
!SkyFlash = $F765
;!UnusedG1 = $F769
!WS_Green = $F76D
;!UnusedG2 = $F771
!Blue_BG_ = $F775
!SpoSpoBG = $F779
!Purp_BG_ = $F77D
!Beacon__ = $F781
!Nor_Hot1 = $F785
!Nor_Hot2 = $F789
!Nor_Hot3 = $F78D
!Nor_Hot4 = $F791
!SandFlor = $F795
!HevySand = $F799
!Waterfal = $F79D
!Tourian1 = $F7A1
!Tourian2 = $F7A5
!Tor_1Esc = $FFC9
!Tor_2Esc = $FFCD
!Tor_3Esc = $FFD1
!Tor_4Esc = $FFD5
!OldT1Esc = $FFD9
!OldT2Esc = $FFDD
!OldT3Esc = $FFE1
!SurfcEsc = $FFE5
!Sky_Esc_ = $FFE9
!CRE_Esc_ = $FFED

  ;  01         02         04         08         10         20         40         80
Glow_Area_0a:
  DW !SkyFlash, !SurfcEsc, !Sky_Esc_, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_0b:
  DW !OldT3Esc, !NullGlow, !NullGlow, !OldT1Esc, !OldT2Esc, !SandFlor, !HevySand, !SamusHot
Glow_Area_1:
  DW !Blue_BG_, !Purp_BG_, !Beacon__, !SpoSpoBG, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_2:
  DW !NullGlow, !Nor_Hot1, !Nor_Hot2, !Nor_Hot3, !Nor_Hot4, !SandFlor, !HevySand, !SamusHot
Glow_Area_3:
  DW !WS_Green, !NullGlow, !NullGlow, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_4:
  DW !NullGlow, !NullGlow, !Waterfal, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_5:
  DW !Tor_4Esc, !Tourian1, !Tor_3Esc, !Tor_1Esc, !Tor_2Esc, !SandFlor, !HevySand, !SamusHot
Glow_Area_6:
  DW !NullGlow, !NullGlow, !NullGlow, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_7:
  DW !NullGlow, !NullGlow, !NullGlow, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot


!NullAnim = $824B
!V_Spike_ = $8251
!H_Spike_ = $8257
!Ocean___ = $825D
;!UnusedA1 = $8263
;!UnusedA2 = $8269
!Laundry_ = $826F
!R_Tread_ = $8275
!L_Tread_ = $827B
!VilePlum = $8281
!SandHead = $8287
!SandFall = $828D

  ;  01         02         04         08         10         20         40         80
Anim_Area_0:
  DW !H_Spike_, !V_Spike_, !Ocean___, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_1:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_2:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_3:
  DW !H_Spike_, !V_Spike_, !Laundry_, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_4:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, !SandHead, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_5:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_6:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_
Anim_Area_7:
  DW !H_Spike_, !V_Spike_, !NullAnim, !SandFall, #Ceiling_, !VilePlum, !R_Tread_, !L_Tread_

GlowTypeTable:
  DW Glow_Area_0a, Glow_Area_0a ;Crateria Surface
  DW Glow_Area_0b, Glow_Area_0b ;Inner Crateria
  DW Glow_Area_3, Glow_Area_3 ;Wrecked Ship
  DW Glow_Area_1, Glow_Area_1 ;Brinstar
  DW Glow_Area_1 ;Tourian Statues Access
  DW Glow_Area_2, Glow_Area_2 ;Norfair
  DW Glow_Area_4, Glow_Area_4 ;Maridia
  DW Glow_Area_5, Glow_Area_5 ;Tourian
  DW Glow_Area_6, Glow_Area_6, Glow_Area_6, Glow_Area_6, Glow_Area_6, Glow_Area_6 ;Ceres
  DW Glow_Area_7, Glow_Area_7, Glow_Area_7, Glow_Area_7, Glow_Area_7 ;Utility Rooms
  ;Bosses
  DW Glow_Area_1 ;Kraid
  DW Glow_Area_2 ;Crocomire
  DW Glow_Area_4 ;Draygon
  DW Glow_Area_1 ;SpoSpo
  DW Glow_Area_3 ;Phantoon
  
AnimTypeTable:
  DW Anim_Area_0, Anim_Area_0 ;Crateria Surface
  DW Anim_Area_0, Anim_Area_0 ;Inner Crateria
  DW Anim_Area_3, Anim_Area_3 ;Wrecked Ship
  DW Anim_Area_1, Anim_Area_1 ;Brinstar
  DW Anim_Area_1 ;Tourian Statues Access
  DW Anim_Area_2, Anim_Area_2 ;Norfair
  DW Anim_Area_4, Anim_Area_4 ;Maridia
  DW Anim_Area_5, Anim_Area_5 ;Tourian
  DW Anim_Area_6, Anim_Area_6, Anim_Area_6, Anim_Area_6, Anim_Area_6, Anim_Area_6 ;Ceres
  DW Anim_Area_0, Anim_Area_0, Anim_Area_0, Anim_Area_0, Anim_Area_0 ;Utility Rooms
  ;Bosses
  DW Anim_Area_1 ;Kraid
  DW Anim_Area_2 ;Crocomire
  DW Anim_Area_4 ;Draygon
  DW Anim_Area_1 ;SpoSpo
  DW Anim_Area_3 ;Phantoon

warnpc $83BA00