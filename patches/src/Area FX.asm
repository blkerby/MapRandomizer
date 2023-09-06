lorom


bank $83 

; Use tileset index to figure out glows instead of area
org $89AC62
  JSR GetTilesetIndex ;LDA $079F
  ASL
  TAY
  LDA GlowTypeTable,Y ;DB is $83

org $89AC98
  JSR GetTilesetIndex ;LDA $079F
  ASL
  TAY
  LDA AnimTypeTable,Y

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

bank auto

CallGlowHandler:
  PHB
  PHK
  PLB
  JSR GetTilesetIndex
  PHX
  ASL
  TAX
  JSR (GlowHandlerTable,x)
  PLX
  PLB
  RTS

bank $83

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
  LDA GlowTypeTable,Y ;DB is $83
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

bank auto

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
  DB $19 ;Inner Crateria
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
  DW !SamusHot, !Nor_Hot1, !Nor_Hot2, !Nor_Hot3, !Nor_Hot4, !SandFlor, !HevySand, !SamusHot
Glow_Area_3:
  DW !WS_Green, !NullGlow, !NullGlow, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
Glow_Area_4:
  DW !SandFlor, !HevySand, !Waterfal, !NullGlow, !NullGlow, !SandFlor, !HevySand, !SamusHot
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
