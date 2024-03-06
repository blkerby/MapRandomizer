lorom

; Support for custom area palette glows
; Handles tileset glows that would be misplaced when the tileset's area and the map's area don't match
; Extends the glow header format to be polumorphic. The vanilla header type is V1, and the new one is V2.
; V1 headers look like this:
; org $8DXXXX
;   DW GlowInitCode, GlowInstructionList
; org $8DYYYY
; GlowInitCode:
; org $8DZZZZ
; GlowInstructionList:
;
; V2 headers lok like this:
; org 8DXXXX
;   DW $00BB, $TTTT
; org $BBTTTT
;   DW GlowInitCode_0, GlowInstructionList_0
; ...
;   DW GlowInitCode_7, GlowInstructionList_7
; org $BBYYY0
; GlowInitCode_0:
; org $BBZZZ0
; GlowInstructionList_0:
; ...
; org $BBYYY7
; GlowInitCode_7:
; org $BBZZZ7
; GlowInstructionList_7:
;
; During each frame, the correct main handler is called for each glow for V1 or V2.
; All code for instructions for V2 glows is run in bank BB, so any instructions used need to be copied there.
; It's not possible to mix banks for a single glow. If a glow is converted to V2, it must be entirely moved.

org $8DC4FF ; Patch main glow object constructor to support polymorphic glow headers
  TYA
  STA $1E7D,X
  STZ $1E8D,X
  LDA #$0001
  STA $1ECD,X
  STZ $1EDD,X

  LDA $0000,Y
  BMI +
  JMP.l SpawnGlow_V2
+
  JMP.w SpawnGlow_V1
MainGlowHandler_ToV2:
  JSL MainGlowHandler_V2
  BRA MainGlowHandler_Continue
warnpc $8DC528

org $8DC527
MainGlowHandler:
  PHP
  PHB
  PHK
  PLB
  REP #$30
  BIT $1E79
  BPL MainGlowHandler_Exit
  LDX #$000E
MainGlowHandler_Loop:
  STX $1E7B
  LDA $1E7D,X
  BEQ MainGlowHandler_Continue
  BMI +
  BRA MainGlowHandler_ToV2
+
  JSR MainGlowHandler_V1
MainGlowHandler_Continue:
  DEX
  DEX
  BPL MainGlowHandler_Loop
MainGlowHandler_Exit:
  PLB
  PLP
  RTL

org $8DC54C
MainGlowHandler_V1:

org $8DF891 ; overwrite moved glow
SpawnGlow_V1:
  LDA #$C594  ; points to an RTS
  STA $1EAD,X ; pre-instruction
  LDA $0002,Y
  STA $1EBD,X
  TXA
  TYX
  TAY
  JSR ($0000,X)
  PLX
  PLB
  PLP
  CLC
  RTL

org $BB8000
SpawnGlow_V2:
  CMP #$00BB ; assert tag
  BEQ +
  JSL $808573 ; crash
+
  STA $1E7D,X

  LDA.w #EmptyPre
  STA $1EAD,X ; pre-instruction

  LDA $1F5B ; map area
  AND #$00FF
  ASL
  ASL ; 4 byte entries
  CLC
  ADC $0002,Y ; add area table base address
  TAY ; points to the header for the glow for the area in bank BB
  PHK
  PLB

  LDA $0002,Y
  STA $1EBD,X
  TXA
  TYX
  TAY
  JSR ($0000,X)
  PLX
  PLB
  PLP
  CLC
  RTL

EmptyInit:
EmptyPre:
  RTS

MainGlowHandler_V2:
  PHB
  PHK
  PLB
  JSR ($1EAD,X)
  LDX $1E7B
  DEC $1ECD,X
  BNE MainGlowHandler_V2_Exit
  LDA $1EBD,X
  TAY
MainGlowHandler_V2_Loop1:
  LDA $0000,Y
  BPL MainGlowHandler_V2_Break1
  STA $12
  INY
  INY
  PEA.w MainGlowHandler_V2_Loop1-1
  JMP ($0012)
MainGlowHandler_V2_Break1:
  STA $1ECD,X
  LDA $1E8D,X
  TAX
MainGlowHandler_V2_Loop2:
  LDA $0002,Y
  BPL MainGlowHandler_V2_Color
  STA $12
  PEA.w MainGlowHandler_V2_Loop2-1
  JMP ($0012)
MainGlowHandler_V2_Color:
  STA $7EC000,X
  INX
  INX
  INY
  INY
  BRA MainGlowHandler_V2_Loop2

GlowYeild: ; C595
  PLA
  LDX $1E7B
  TYA
  CLC
  ADC #$0004
  STA $1EBD,X
MainGlowHandler_V2_Exit:
  PLB
  RTL

GlowJMP: ; C61E
  LDA $0000,Y
  TAY
  RTS

SetLoopCounter: ; C648
  SEP #$20
  LDA $0000,Y
  STA $1EDD,X
  REP #$20
  INY
  RTS

DecAndLoop: ; C639
  DEC $1EDD,X
  BNE GlowJMP
  INY
  INY
  RTS

GlowDelete: ; C5CF
  STZ $1E7D,X
  PLA
  RTS

SetLinkTarget:
  LDA $0000,Y
  STA $1E9D,X
  INY
  INY
  RTS

LinkJMP:
  LDA $1E9D,X
  TAY
  RTS

SetPreInstruction: ; C5D4
  LDA $0000,Y
  STA $1EAD,X
  INY
  INY
  RTS

SetColorIndex: ; C655
  LDA $0000,Y
  STA $1E8D,X
  INY
  INY
  RTS

SkipColors_2: ; C599
  TXA
  CLC
  ADC #$0004
  TAX
  INY
  INY
  RTS

SkipColors_3: ; C5A2
  TXA
  CLC
  ADC #$0006
  TAX
  INY
  INY
  RTS

SkipColors_4: ; C5AB
  TXA
  CLC
  ADC #$0008
  TAX
  INY
  INY
  RTS

SkipColors_8: ; C5B4
  TXA
  CLC
  ADC #$0010
  TAX
  INY
  INY
  RTS

SkipColors_9: ; C5BD
  TXA
  CLC
  ADC #$0012
  TAX
  INY
  INY
  RTS

PlaySFX: ; C673
  LDA $0000,Y
  JSL $8090CB
  INY
  RTS

; Crateria tileset glows

ResetLightning:
  LDA $0AFA
  CMP $197A
  BCS +
  LDA #$0001
  STA $1ECD,X
  LDA $1E9D,X
  STA $1EBD,X
+
  RTS

SkyFlashTable:
  DW EmptyInit,SkyFlash0_List, EmptyInit,SkyFlash1_List, EmptyInit,SkyFlash2_List, EmptyInit,SkyFlash3_List
  DW EmptyInit,SkyFlash4_List, EmptyInit,SkyFlash5_List, EmptyInit,SkyFlash6_List, EmptyInit,SkyFlash7_List

!SkyFlash0_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash0_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash0_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash0_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash0_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash1_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash1_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash1_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash1_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash1_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash2_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash2_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash2_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash2_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash2_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash3_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash3_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash3_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash3_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash3_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash4_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash4_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash4_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash4_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash4_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash5_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash5_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash5_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash5_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash5_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash6_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash6_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash6_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash6_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash6_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

!SkyFlash7_Colors_0 = $2D6C, $294B, $252A, $2109, $1CE8, $18C7, $14A6, $1085
!SkyFlash7_Colors_1 = $4632, $4211, $3DF0, $39CF, $35AE, $318D, $2D6C, $294B
!SkyFlash7_Colors_2 = $5EF8, $5AD7, $56B6, $5295, $4E74, $4A53, $4632, $4211
!SkyFlash7_Colors_3 = $77BE, $739D, $6F7C, $6B5B, $673A, $6319, $5EF8, $5AD7
!SkyFlash7_Colors_4 = $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF, $7FFF

macro SkyFlash_List(n)
SkyFlash<n>_List:
  DW SetLinkTarget, SkyFlash<n>_List_Loop1
  DW SetPreInstruction, ResetLightning
  DW SetColorIndex, $00A8
SkyFlash<n>_List_Loop1:
  DW $00F0
    DW !SkyFlash<n>_Colors_0
    DW GlowYeild
  DW SetLoopCounter : DB $02
SkyFlash<n>_List_Loop2:
  DW $0002
    DW !SkyFlash<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_4
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_2
    DW GlowYeild
  DW $0002
    DW !SkyFlash<n>_Colors_1
    DW GlowYeild
  DW DecAndLoop, SkyFlash<n>_List_Loop2
  DW $00F0
    DW !SkyFlash<n>_Colors_0
    DW GlowYeild
  DW SetLoopCounter : DB $01
SkyFlash<n>_List_Loop3:
  DW $0001
    DW !SkyFlash<n>_Colors_4
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !SkyFlash<n>_Colors_2
    DW GlowYeild
  DW $0002
    DW !SkyFlash<n>_Colors_1
    DW GlowYeild
  DW DecAndLoop, SkyFlash<n>_List_Loop3
  DW GlowJMP, SkyFlash<n>_List_Loop1
endmacro

%SkyFlash_List(0)
%SkyFlash_List(1)
%SkyFlash_List(2)
%SkyFlash_List(3)
%SkyFlash_List(4)
%SkyFlash_List(5)
%SkyFlash_List(6)
%SkyFlash_List(7)

SurfcEscTable:
  DW EmptyInit,SurfcEsc0_List, EmptyInit,SurfcEsc1_List, EmptyInit,SurfcEsc2_List, EmptyInit,SurfcEsc3_List
  DW EmptyInit,SurfcEsc4_List, EmptyInit,SurfcEsc5_List, EmptyInit,SurfcEsc6_List, EmptyInit,SurfcEsc7_List

!SurfcEsc0_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc0_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc0_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc0_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc0_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc0_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc0_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc0_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc1_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc1_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc1_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc1_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc1_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc1_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc1_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc1_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc2_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc2_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc2_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc2_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc2_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc2_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc2_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc2_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc3_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc3_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc3_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc3_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc3_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc3_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc3_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc3_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc4_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc4_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc4_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc4_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc4_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc4_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc4_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc4_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc5_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc5_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc5_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc5_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc5_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc5_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc5_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc5_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc6_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc6_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc6_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc6_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc6_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc6_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc6_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc6_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

!SurfcEsc7_Colors_0 = $1D89, $0D06, $0CA3, $2D0E, $2D09, $28C5, $0C81
!SurfcEsc7_Colors_1 = $1D8A, $0D07, $0CA4, $2D0E, $2D09, $28C5, $0421
!SurfcEsc7_Colors_2 = $1D8A, $0D28, $0CC4, $2D0F, $2D0A, $28C6, $0423
!SurfcEsc7_Colors_3 = $1D8B, $0D29, $0CC5, $2D0F, $2D0A, $28C6, $0424
!SurfcEsc7_Colors_4 = $1DAB, $1149, $10C5, $2D0F, $2D0B, $28C7, $0845
!SurfcEsc7_Colors_5 = $1DAC, $114A, $10C6, $2D0F, $2D0B, $28C7, $0846
!SurfcEsc7_Colors_6 = $1DAC, $116B, $10E6, $2D10, $2D0C, $28C8, $0848
!SurfcEsc7_Colors_7 = $1DAD, $116C, $10E7, $2D10, $2D0C, $28C8, $0015

macro SurfcEsc_List(n)
SurfcEsc<n>_List:
  DW SetColorIndex, $0082
SurfcEsc<n>_List_Loop:
  DW $0008
    DW !SurfcEsc<n>_Colors_0
    DW GlowYeild
  DW $0007
    DW !SurfcEsc<n>_Colors_1
    DW GlowYeild
  DW $0006
    DW !SurfcEsc<n>_Colors_2
    DW GlowYeild
  DW $0005
    DW !SurfcEsc<n>_Colors_3
    DW GlowYeild
  DW $0004
    DW !SurfcEsc<n>_Colors_4
    DW GlowYeild
  DW $0003
    DW !SurfcEsc<n>_Colors_5
    DW GlowYeild
  DW $0002
    DW !SurfcEsc<n>_Colors_6
    DW GlowYeild
  DW $0001
    DW !SurfcEsc<n>_Colors_7
    DW GlowYeild
  DW $0002
    DW !SurfcEsc<n>_Colors_6
    DW GlowYeild
  DW $0003
    DW !SurfcEsc<n>_Colors_5
    DW GlowYeild
  DW $0004
    DW !SurfcEsc<n>_Colors_4
    DW GlowYeild
  DW $0005
    DW !SurfcEsc<n>_Colors_3
    DW GlowYeild
  DW $0006
    DW !SurfcEsc<n>_Colors_2
    DW GlowYeild
  DW $0007
    DW !SurfcEsc<n>_Colors_1
    DW GlowYeild
  DW GlowJMP, SurfcEsc<n>_List_Loop
endmacro

%SurfcEsc_List(0)
%SurfcEsc_List(1)
%SurfcEsc_List(2)
%SurfcEsc_List(3)
%SurfcEsc_List(4)
%SurfcEsc_List(5)
%SurfcEsc_List(6)
%SurfcEsc_List(7)

Sky_Esc_Table:
  DW EmptyInit,Sky_Esc_0_List, EmptyInit,Sky_Esc_1_List, EmptyInit,Sky_Esc_2_List, EmptyInit,Sky_Esc_3_List
  DW EmptyInit,Sky_Esc_4_List, EmptyInit,Sky_Esc_5_List, EmptyInit,Sky_Esc_6_List, EmptyInit,Sky_Esc_7_List

!Sky_Esc_0_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_0_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_0_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_0_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_1_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_1_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_1_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_1_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_2_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_2_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_2_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_2_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_3_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_3_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_3_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_3_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_4_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_4_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_4_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_4_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_5_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_5_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_5_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_5_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_6_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_6_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_6_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_6_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

!Sky_Esc_7_Colors_0 = $48D5, $38B0, $286A, $2488, $2067, $1846, $1425, $1024, $0C23, $0C03, $0802
!Sky_Esc_7_Colors_1 = $3DD8, $31D5, $2991, $25B0, $218F, $1D8E, $0C23, $0C23, $0822, $0802, $0401
!Sky_Esc_7_Colors_2 = $32FC, $2EDA, $26D8, $26D7, $26D7, $22B7, $0802, $0401, $0401, $0401, $0401
!Sky_Esc_7_Colors_3 = $27FF, $27FF, $27FF, $27FF, $27FF, $27FF, $0000, $0000, $0401, $0000, $0000

macro Sky_Esc_List(n)
Sky_Esc_<n>_List:
  DW SetColorIndex, $00A2
Sky_Esc_<n>_List_Loop:
  DW $0031
    DW !Sky_Esc_<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_2
    DW GlowYeild
  DW $0011
    DW !Sky_Esc_<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_3
    DW GlowYeild
  DW $0018
    DW !Sky_Esc_<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !Sky_Esc_<n>_Colors_2
    DW GlowYeild
  DW GlowJMP, Sky_Esc_<n>_List_Loop
endmacro

%Sky_Esc_List(0)
%Sky_Esc_List(1)
%Sky_Esc_List(2)
%Sky_Esc_List(3)
%Sky_Esc_List(4)
%Sky_Esc_List(5)
%Sky_Esc_List(6)
%Sky_Esc_List(7)

OldT1EscTable:
  DW EmptyInit,OldT1Esc0_List, EmptyInit,OldT1Esc1_List, EmptyInit,OldT1Esc2_List, EmptyInit,OldT1Esc3_List
  DW EmptyInit,OldT1Esc4_List, EmptyInit,OldT1Esc5_List, EmptyInit,OldT1Esc6_List, EmptyInit,OldT1Esc7_List

!OldT1Esc0_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc0_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc0_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc0_Colors_3a = $3576, $28EF, $1889
!OldT1Esc0_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc0_Colors_5a = $20B8, $1891, $106A
!OldT1Esc0_Colors_6a = $1479, $1051, $084A
!OldT1Esc0_Colors_7a = $081A, $0812, $042B
!OldT1Esc0_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc0_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc0_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc0_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc0_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc0_Colors_0c = $7F9C
!OldT1Esc0_Colors_1c = $6F3C
!OldT1Esc0_Colors_2c = $62FD
!OldT1Esc0_Colors_3c = $529D
!OldT1Esc0_Colors_4c = $423E
!OldT1Esc0_Colors_5c = $31DE
!OldT1Esc0_Colors_6c = $259F
!OldT1Esc0_Colors_7c = $153F

!OldT1Esc1_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc1_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc1_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc1_Colors_3a = $3576, $28EF, $1889
!OldT1Esc1_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc1_Colors_5a = $20B8, $1891, $106A
!OldT1Esc1_Colors_6a = $1479, $1051, $084A
!OldT1Esc1_Colors_7a = $081A, $0812, $042B
!OldT1Esc1_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc1_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc1_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc1_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc1_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc1_Colors_0c = $7F9C
!OldT1Esc1_Colors_1c = $6F3C
!OldT1Esc1_Colors_2c = $62FD
!OldT1Esc1_Colors_3c = $529D
!OldT1Esc1_Colors_4c = $423E
!OldT1Esc1_Colors_5c = $31DE
!OldT1Esc1_Colors_6c = $259F
!OldT1Esc1_Colors_7c = $153F

!OldT1Esc2_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc2_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc2_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc2_Colors_3a = $3576, $28EF, $1889
!OldT1Esc2_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc2_Colors_5a = $20B8, $1891, $106A
!OldT1Esc2_Colors_6a = $1479, $1051, $084A
!OldT1Esc2_Colors_7a = $081A, $0812, $042B
!OldT1Esc2_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc2_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc2_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc2_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc2_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc2_Colors_0c = $7F9C
!OldT1Esc2_Colors_1c = $6F3C
!OldT1Esc2_Colors_2c = $62FD
!OldT1Esc2_Colors_3c = $529D
!OldT1Esc2_Colors_4c = $423E
!OldT1Esc2_Colors_5c = $31DE
!OldT1Esc2_Colors_6c = $259F
!OldT1Esc2_Colors_7c = $153F

!OldT1Esc3_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc3_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc3_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc3_Colors_3a = $3576, $28EF, $1889
!OldT1Esc3_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc3_Colors_5a = $20B8, $1891, $106A
!OldT1Esc3_Colors_6a = $1479, $1051, $084A
!OldT1Esc3_Colors_7a = $081A, $0812, $042B
!OldT1Esc3_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc3_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc3_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc3_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc3_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc3_Colors_0c = $7F9C
!OldT1Esc3_Colors_1c = $6F3C
!OldT1Esc3_Colors_2c = $62FD
!OldT1Esc3_Colors_3c = $529D
!OldT1Esc3_Colors_4c = $423E
!OldT1Esc3_Colors_5c = $31DE
!OldT1Esc3_Colors_6c = $259F
!OldT1Esc3_Colors_7c = $153F

!OldT1Esc4_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc4_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc4_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc4_Colors_3a = $3576, $28EF, $1889
!OldT1Esc4_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc4_Colors_5a = $20B8, $1891, $106A
!OldT1Esc4_Colors_6a = $1479, $1051, $084A
!OldT1Esc4_Colors_7a = $081A, $0812, $042B
!OldT1Esc4_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc4_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc4_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc4_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc4_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc4_Colors_0c = $7F9C
!OldT1Esc4_Colors_1c = $6F3C
!OldT1Esc4_Colors_2c = $62FD
!OldT1Esc4_Colors_3c = $529D
!OldT1Esc4_Colors_4c = $423E
!OldT1Esc4_Colors_5c = $31DE
!OldT1Esc4_Colors_6c = $259F
!OldT1Esc4_Colors_7c = $153F

!OldT1Esc5_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc5_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc5_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc5_Colors_3a = $3576, $28EF, $1889
!OldT1Esc5_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc5_Colors_5a = $20B8, $1891, $106A
!OldT1Esc5_Colors_6a = $1479, $1051, $084A
!OldT1Esc5_Colors_7a = $081A, $0812, $042B
!OldT1Esc5_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc5_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc5_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc5_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc5_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc5_Colors_0c = $7F9C
!OldT1Esc5_Colors_1c = $6F3C
!OldT1Esc5_Colors_2c = $62FD
!OldT1Esc5_Colors_3c = $529D
!OldT1Esc5_Colors_4c = $423E
!OldT1Esc5_Colors_5c = $31DE
!OldT1Esc5_Colors_6c = $259F
!OldT1Esc5_Colors_7c = $153F

!OldT1Esc6_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc6_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc6_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc6_Colors_3a = $3576, $28EF, $1889
!OldT1Esc6_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc6_Colors_5a = $20B8, $1891, $106A
!OldT1Esc6_Colors_6a = $1479, $1051, $084A
!OldT1Esc6_Colors_7a = $081A, $0812, $042B
!OldT1Esc6_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc6_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc6_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc6_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc6_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc6_Colors_0c = $7F9C
!OldT1Esc6_Colors_1c = $6F3C
!OldT1Esc6_Colors_2c = $62FD
!OldT1Esc6_Colors_3c = $529D
!OldT1Esc6_Colors_4c = $423E
!OldT1Esc6_Colors_5c = $31DE
!OldT1Esc6_Colors_6c = $259F
!OldT1Esc6_Colors_7c = $153F

!OldT1Esc7_Colors_0a = $5A73, $41AD, $28E7
!OldT1Esc7_Colors_1a = $4E14, $396E, $24C8
!OldT1Esc7_Colors_2a = $41D5, $312E, $1CA8
!OldT1Esc7_Colors_3a = $3576, $28EF, $1889
!OldT1Esc7_Colors_4a = $2D17, $20D0, $1489
!OldT1Esc7_Colors_5a = $20B8, $1891, $106A
!OldT1Esc7_Colors_6a = $1479, $1051, $084A
!OldT1Esc7_Colors_7a = $081A, $0812, $042B
!OldT1Esc7_Colors_0b = $0019, $0012, $3460, $0C20
!OldT1Esc7_Colors_1b = $0014, $000E, $4900, $1C60
!OldT1Esc7_Colors_2b = $000F, $000A, $5980, $2CA0
!OldT1Esc7_Colors_3b = $000A, $0005, $6E20, $38C0
!OldT1Esc7_Colors_4b = $0005, $0001, $7EA0, $4900
!OldT1Esc7_Colors_0c = $7F9C
!OldT1Esc7_Colors_1c = $6F3C
!OldT1Esc7_Colors_2c = $62FD
!OldT1Esc7_Colors_3c = $529D
!OldT1Esc7_Colors_4c = $423E
!OldT1Esc7_Colors_5c = $31DE
!OldT1Esc7_Colors_6c = $259F
!OldT1Esc7_Colors_7c = $153F

macro OldT1Esc_List(n)
OldT1Esc<n>_List:
  DW SetColorIndex, $00A2
OldT1Esc<n>_List_Loop:
  DW $0003
    DW !OldT1Esc<n>_Colors_0a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_0b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_0c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_1a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_1b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_1c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_2a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_2b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_2c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_3a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_3b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_3c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_4a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_4b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_4c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_5a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_3b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_5c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_6a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_2b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_6c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_7a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_1b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_7c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_6a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_2b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_6c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_5a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_3b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_5c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_4a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_4b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_4c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_3a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_3b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_3c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_2a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_2b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_2c
    DW GlowYeild
  DW $0003
    DW !OldT1Esc<n>_Colors_1a
    DW SkipColors_4
    DW !OldT1Esc<n>_Colors_1b
    DW SkipColors_2
    DW !OldT1Esc<n>_Colors_1c
    DW GlowYeild
  DW GlowJMP, OldT1Esc<n>_List_Loop
endmacro

%OldT1Esc_List(0)
%OldT1Esc_List(1)
%OldT1Esc_List(2)
%OldT1Esc_List(3)
%OldT1Esc_List(4)
%OldT1Esc_List(5)
%OldT1Esc_List(6)
%OldT1Esc_List(7)

OldT2EscTable:
  DW EmptyInit,OldT2Esc0_List, EmptyInit,OldT2Esc1_List, EmptyInit,OldT2Esc2_List, EmptyInit,OldT2Esc3_List
  DW EmptyInit,OldT2Esc4_List, EmptyInit,OldT2Esc5_List, EmptyInit,OldT2Esc6_List, EmptyInit,OldT2Esc7_List

!OldT2Esc0_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc0_Colors_1 = $29D0, $150A, $0885
!OldT2Esc0_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc0_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc0_Colors_4 = $025A, $0192, $00CA

!OldT2Esc1_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc1_Colors_1 = $29D0, $150A, $0885
!OldT2Esc1_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc1_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc1_Colors_4 = $025A, $0192, $00CA

!OldT2Esc2_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc2_Colors_1 = $29D0, $150A, $0885
!OldT2Esc2_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc2_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc2_Colors_4 = $025A, $0192, $00CA

!OldT2Esc3_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc3_Colors_1 = $29D0, $150A, $0885
!OldT2Esc3_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc3_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc3_Colors_4 = $025A, $0192, $00CA

!OldT2Esc4_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc4_Colors_1 = $29D0, $150A, $0885
!OldT2Esc4_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc4_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc4_Colors_4 = $025A, $0192, $00CA

!OldT2Esc5_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc5_Colors_1 = $29D0, $150A, $0885
!OldT2Esc5_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc5_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc5_Colors_4 = $025A, $0192, $00CA

!OldT2Esc6_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc6_Colors_1 = $29D0, $150A, $0885
!OldT2Esc6_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc6_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc6_Colors_4 = $025A, $0192, $00CA

!OldT2Esc7_Colors_0 = $35AD, $1CE7, $0C63
!OldT2Esc7_Colors_1 = $29D0, $150A, $0885
!OldT2Esc7_Colors_2 = $1E14, $114D, $08A7
!OldT2Esc7_Colors_3 = $0E37, $096F, $04A8
!OldT2Esc7_Colors_4 = $025A, $0192, $00CA

macro OldT2Esc_List(n)
OldT2Esc<n>_List:
  DW SetColorIndex, $00D2
OldT2Esc<n>_List_Loop:
  DW $0010
    DW !OldT2Esc<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_2
    DW GlowYeild
  DW $0002
    DW !OldT2Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_4
    DW GlowYeild
  DW $0002
    DW !OldT2Esc<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_4
    DW GlowYeild
  DW $0020
    DW !OldT2Esc<n>_Colors_0
    DW GlowYeild
  DW $0002
    DW !OldT2Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT2Esc<n>_Colors_4
    DW GlowYeild
  DW GlowJMP, OldT2Esc<n>_List_Loop
endmacro

%OldT2Esc_List(0)
%OldT2Esc_List(1)
%OldT2Esc_List(2)
%OldT2Esc_List(3)
%OldT2Esc_List(4)
%OldT2Esc_List(5)
%OldT2Esc_List(6)
%OldT2Esc_List(7)

OldT3EscTable:
  DW EmptyInit,OldT3Esc0_List, EmptyInit,OldT3Esc1_List, EmptyInit,OldT3Esc2_List, EmptyInit,OldT3Esc3_List
  DW EmptyInit,OldT3Esc4_List, EmptyInit,OldT3Esc5_List, EmptyInit,OldT3Esc6_List, EmptyInit,OldT3Esc7_List

!OldT3Esc0_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc0_Colors_1 = $398E, $296B, $1549
!OldT3Esc0_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc0_Colors_3 = $5739, $3318, $0B18
!OldT3Esc0_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc1_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc1_Colors_1 = $398E, $296B, $1549
!OldT3Esc1_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc1_Colors_3 = $5739, $3318, $0B18
!OldT3Esc1_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc2_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc2_Colors_1 = $398E, $296B, $1549
!OldT3Esc2_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc2_Colors_3 = $5739, $3318, $0B18
!OldT3Esc2_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc3_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc3_Colors_1 = $398E, $296B, $1549
!OldT3Esc3_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc3_Colors_3 = $5739, $3318, $0B18
!OldT3Esc3_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc4_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc4_Colors_1 = $398E, $296B, $1549
!OldT3Esc4_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc4_Colors_3 = $5739, $3318, $0B18
!OldT3Esc4_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc5_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc5_Colors_1 = $398E, $296B, $1549
!OldT3Esc5_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc5_Colors_3 = $5739, $3318, $0B18
!OldT3Esc5_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc6_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc6_Colors_1 = $398E, $296B, $1549
!OldT3Esc6_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc6_Colors_3 = $5739, $3318, $0B18
!OldT3Esc6_Colors_4 = $67FF, $43FF, $03FF

!OldT3Esc7_Colors_0 = $28C8, $2484, $1C61
!OldT3Esc7_Colors_1 = $398E, $296B, $1549
!OldT3Esc7_Colors_2 = $4A74, $2E52, $1230
!OldT3Esc7_Colors_3 = $5739, $3318, $0B18
!OldT3Esc7_Colors_4 = $67FF, $43FF, $03FF

macro OldT3Esc_List(n)
OldT3Esc<n>_List:
  DW SetColorIndex, $00AA
OldT3Esc<n>_List_Loop:
  DW $0010
    DW !OldT3Esc<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_2
    DW GlowYeild
  DW $0002
    DW !OldT3Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_4
    DW GlowYeild
  DW $0002
    DW !OldT3Esc<n>_Colors_0
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_4
    DW GlowYeild
  DW $0020
    DW !OldT3Esc<n>_Colors_0
    DW GlowYeild
  DW $0002
    DW !OldT3Esc<n>_Colors_1
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_2
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_3
    DW GlowYeild
  DW $0001
    DW !OldT3Esc<n>_Colors_4
    DW GlowYeild
  DW GlowJMP, OldT3Esc<n>_List_Loop
endmacro

%OldT3Esc_List(0)
%OldT3Esc_List(1)
%OldT3Esc_List(2)
%OldT3Esc_List(3)
%OldT3Esc_List(4)
%OldT3Esc_List(5)
%OldT3Esc_List(6)
%OldT3Esc_List(7)

; Brinstar tileset glows
; 94 75 23
; 60 49 8
Blue_BG_Table:
  DW EmptyInit,Blue_BG_0_List, EmptyInit,Blue_BG_1_List, EmptyInit,Blue_BG_2_List, EmptyInit,Blue_BG_3_List
  DW EmptyInit,Blue_BG_4_List, EmptyInit,Blue_BG_5_List, EmptyInit,Blue_BG_6_List, EmptyInit,Blue_BG_7_List

!Blue_BG_0_Colors_0 = $584A, $3827, $1803
!Blue_BG_0_Colors_1 = $5449, $3426, $1403
!Blue_BG_0_Colors_2 = $5049, $3026, $1402
!Blue_BG_0_Colors_3 = $4C48, $2C25, $1002
!Blue_BG_0_Colors_4 = $4428, $2C05, $1002
!Blue_BG_0_Colors_5 = $4027, $2804, $0C02
!Blue_BG_0_Colors_6 = $3C27, $2404, $0C01
!Blue_BG_0_Colors_7 = $3826, $2003, $0801

!Blue_BG_1_Colors_0 = $5D22, $4463, $1840
!Blue_BG_1_Colors_1 = $5901, $4042, $1420
!Blue_BG_1_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_1_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_1_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_1_Colors_5 = $4880, $3800, $0C00
!Blue_BG_1_Colors_6 = $4460, $3400, $0800
!Blue_BG_1_Colors_7 = $4040, $3400, $0800

!Blue_BG_2_Colors_0 = $5D22, $4463, $1840
!Blue_BG_2_Colors_1 = $5901, $4042, $1420
!Blue_BG_2_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_2_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_2_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_2_Colors_5 = $4880, $3800, $0C00
!Blue_BG_2_Colors_6 = $4460, $3400, $0800
!Blue_BG_2_Colors_7 = $4040, $3400, $0800

!Blue_BG_3_Colors_0 = $5D22, $4463, $1840
!Blue_BG_3_Colors_1 = $5901, $4042, $1420
!Blue_BG_3_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_3_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_3_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_3_Colors_5 = $4880, $3800, $0C00
!Blue_BG_3_Colors_6 = $4460, $3400, $0800
!Blue_BG_3_Colors_7 = $4040, $3400, $0800

!Blue_BG_4_Colors_0 = $5D22, $4463, $1840
!Blue_BG_4_Colors_1 = $5901, $4042, $1420
!Blue_BG_4_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_4_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_4_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_4_Colors_5 = $4880, $3800, $0C00
!Blue_BG_4_Colors_6 = $4460, $3400, $0800
!Blue_BG_4_Colors_7 = $4040, $3400, $0800

!Blue_BG_5_Colors_0 = $5D22, $4463, $1840
!Blue_BG_5_Colors_1 = $5901, $4042, $1420
!Blue_BG_5_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_5_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_5_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_5_Colors_5 = $4880, $3800, $0C00
!Blue_BG_5_Colors_6 = $4460, $3400, $0800
!Blue_BG_5_Colors_7 = $4040, $3400, $0800

!Blue_BG_6_Colors_0 = $5D22, $4463, $1840
!Blue_BG_6_Colors_1 = $5901, $4042, $1420
!Blue_BG_6_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_6_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_6_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_6_Colors_5 = $4880, $3800, $0C00
!Blue_BG_6_Colors_6 = $4460, $3400, $0800
!Blue_BG_6_Colors_7 = $4040, $3400, $0800

!Blue_BG_7_Colors_0 = $5D22, $4463, $1840
!Blue_BG_7_Colors_1 = $5901, $4042, $1420
!Blue_BG_7_Colors_2 = $54E0, $3C21, $1000
!Blue_BG_7_Colors_3 = $50C0, $3C21, $1000
!Blue_BG_7_Colors_4 = $4CA0, $3800, $0C00
!Blue_BG_7_Colors_5 = $4880, $3800, $0C00
!Blue_BG_7_Colors_6 = $4460, $3400, $0800
!Blue_BG_7_Colors_7 = $4040, $3400, $0800

macro Blue_BG__List(n)
Blue_BG_<n>_List:
  DW SetColorIndex, $00E2
Blue_BG_<n>_List_Loop:
  DW $000A
    DW !Blue_BG_<n>_Colors_0
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_1
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_4
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_5
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_6
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_7
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_6
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_5
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_4
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !Blue_BG_<n>_Colors_1
    DW GlowYeild
  DW GlowJMP, Blue_BG_<n>_List_Loop
endmacro

%Blue_BG__List(0)
%Blue_BG__List(1)
%Blue_BG__List(2)
%Blue_BG__List(3)
%Blue_BG__List(4)
%Blue_BG__List(5)
%Blue_BG__List(6)
%Blue_BG__List(7)

SpoSpoBGInit:
  PHX
  LDX $079F
  LDA $7ED828,X
  PLX
  AND #$0002
  BEQ +
  LDA #$0000
  STA $1E7D,Y
+
  RTS

SpoSpoBGPreInstruction:
  PHX
  LDX $079F
  LDA $7ED828,X
  PLX
  AND #$0002
  BEQ +
  LDA #$0000
  STA $1E7D,X
+
  RTS

SpoSpoBGTable:
  DW SpoSpoBGInit,SpoSpoBG0_List, SpoSpoBGInit,SpoSpoBG1_List, SpoSpoBGInit,SpoSpoBG2_List, SpoSpoBGInit,SpoSpoBG3_List
  DW SpoSpoBGInit,SpoSpoBG4_List, SpoSpoBGInit,SpoSpoBG5_List, SpoSpoBGInit,SpoSpoBG6_List, SpoSpoBGInit,SpoSpoBG7_List
SpoSpoBG0_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_0_List
SpoSpoBG1_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_1_List
SpoSpoBG2_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_2_List
SpoSpoBG3_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_3_List
SpoSpoBG4_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_4_List
SpoSpoBG5_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_5_List
SpoSpoBG6_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_6_List
SpoSpoBG7_List:
  DW SetPreInstruction, SpoSpoBGPreInstruction
  DW GlowJMP, Blue_BG_7_List

; 87 56 34 26 19 11 8 4
Purp_BG_Table:
  DW EmptyInit,Purp_BG_0_List, EmptyInit,Purp_BG_1_List, EmptyInit,Purp_BG_2_List, EmptyInit,Purp_BG_3_List
  DW EmptyInit,Purp_BG_4_List, EmptyInit,Purp_BG_5_List, EmptyInit,Purp_BG_6_List, EmptyInit,Purp_BG_7_List

!Purp_BG_0_Colors_0 = $544A, $3C09, $2407, $1C06, $1404, $1002, $0C01, $0801
!Purp_BG_0_Colors_1 = $544A, $3C09, $2407, $1C06, $1404, $1002, $0C01, $0801
!Purp_BG_0_Colors_2 = $4C49, $3808, $2006, $1805, $1404, $1002, $0C01, $0801
!Purp_BG_0_Colors_3 = $4448, $3007, $1C06, $1805, $1003, $0C02, $0801, $0801
!Purp_BG_0_Colors_4 = $3C27, $2C07, $1C05, $1404, $1003, $0C01, $0801, $0401
!Purp_BG_0_Colors_5 = $3426, $2806, $1804, $1004, $0C03, $0C01, $0801, $0401
!Purp_BG_0_Colors_6 = $2C25, $2005, $1404, $1003, $0C02, $0801, $0801, $0401
!Purp_BG_0_Colors_7 = $2825, $1C04, $1003, $0C03, $0802, $0801, $0400, $0400

!Purp_BG_1_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_1_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_1_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_1_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_1_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_1_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_1_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_1_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_2_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_2_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_2_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_2_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_2_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_2_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_2_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_2_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_3_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_3_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_3_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_3_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_3_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_3_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_3_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_3_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_4_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_4_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_4_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_4_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_4_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_4_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_4_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_4_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_5_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_5_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_5_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_5_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_5_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_5_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_5_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_5_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_6_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_6_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_6_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_6_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_6_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_6_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_6_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_6_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

!Purp_BG_7_Colors_0 = $4C17, $280F, $2409, $1C07, $1405, $0C03, $0802, $0401
!Purp_BG_7_Colors_1 = $4816, $240E, $2008, $1806, $1004, $0802, $0401, $0000
!Purp_BG_7_Colors_2 = $4415, $200D, $1C07, $1405, $0C03, $0401, $0000, $0000
!Purp_BG_7_Colors_3 = $4014, $1C0C, $1806, $1004, $0802, $0000, $0000, $0000
!Purp_BG_7_Colors_4 = $3C13, $180B, $1405, $0C03, $0401, $0000, $0000, $0000
!Purp_BG_7_Colors_5 = $3812, $140A, $1004, $0802, $0000, $0000, $0000, $0000
!Purp_BG_7_Colors_6 = $3411, $1009, $0C03, $0401, $0000, $0000, $0000, $0000
!Purp_BG_7_Colors_7 = $3010, $0C08, $0802, $0000, $0000, $0000, $0000, $0000

macro Purp_BG__List(n)
Purp_BG_<n>_List:
  DW SetColorIndex, $00C8
Purp_BG_<n>_List_Loop:
  DW $000A
    DW !Purp_BG_<n>_Colors_0
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_1
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_4
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_5
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_6
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_7
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_6
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_5
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_4
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !Purp_BG_<n>_Colors_1
    DW GlowYeild
  DW GlowJMP, Purp_BG_<n>_List_Loop
endmacro

%Purp_BG__List(0)
%Purp_BG__List(1)
%Purp_BG__List(2)
%Purp_BG__List(3)
%Purp_BG__List(4)
%Purp_BG__List(5)
%Purp_BG__List(6)
%Purp_BG__List(7)

; Crateria and Brinstar need to use the same table to keep vanilla rooms working as expected
Beacon__Table:
  DW EmptyInit,Beacon__1_List, EmptyInit,Beacon__1_List, EmptyInit,Beacon__2_List, EmptyInit,Beacon__3_List
  DW EmptyInit,Beacon__4_List, EmptyInit,Beacon__5_List, EmptyInit,Beacon__6_List, EmptyInit,Beacon__7_List

!Beacon__1_Colors_0a = $02BF, $017F, $0015
!Beacon__1_Colors_1a = $023B, $00FB, $0011
!Beacon__1_Colors_2a = $01D8, $0098, $000E
!Beacon__1_Colors_3a = $0154, $0055, $000B
!Beacon__1_Colors_4a = $00D0, $0010, $0007
!Beacon__1_Colors_5a = $00AA, $000B, $0004
!Beacon__1_Colors_0b = $7FFF
!Beacon__1_Colors_1b = $739C
!Beacon__1_Colors_2b = $5AD6
!Beacon__1_Colors_3b = $4E73
!Beacon__1_Colors_4b = $4631
!Beacon__1_Colors_5b = $3DEF

!Beacon__2_Colors_0a = $02BF, $017F, $0015
!Beacon__2_Colors_1a = $023B, $00FB, $0011
!Beacon__2_Colors_2a = $01D8, $0098, $000E
!Beacon__2_Colors_3a = $0154, $0055, $000B
!Beacon__2_Colors_4a = $00D0, $0010, $0007
!Beacon__2_Colors_5a = $00AA, $000B, $0004
!Beacon__2_Colors_0b = $7FFF
!Beacon__2_Colors_1b = $739C
!Beacon__2_Colors_2b = $5AD6
!Beacon__2_Colors_3b = $4E73
!Beacon__2_Colors_4b = $4631
!Beacon__2_Colors_5b = $3DEF

!Beacon__3_Colors_0a = $02BF, $017F, $0015
!Beacon__3_Colors_1a = $023B, $00FB, $0011
!Beacon__3_Colors_2a = $01D8, $0098, $000E
!Beacon__3_Colors_3a = $0154, $0055, $000B
!Beacon__3_Colors_4a = $00D0, $0010, $0007
!Beacon__3_Colors_5a = $00AA, $000B, $0004
!Beacon__3_Colors_0b = $7FFF
!Beacon__3_Colors_1b = $739C
!Beacon__3_Colors_2b = $5AD6
!Beacon__3_Colors_3b = $4E73
!Beacon__3_Colors_4b = $4631
!Beacon__3_Colors_5b = $3DEF

!Beacon__4_Colors_0a = $02BF, $017F, $0015
!Beacon__4_Colors_1a = $023B, $00FB, $0011
!Beacon__4_Colors_2a = $01D8, $0098, $000E
!Beacon__4_Colors_3a = $0154, $0055, $000B
!Beacon__4_Colors_4a = $00D0, $0010, $0007
!Beacon__4_Colors_5a = $00AA, $000B, $0004
!Beacon__4_Colors_0b = $7FFF
!Beacon__4_Colors_1b = $739C
!Beacon__4_Colors_2b = $5AD6
!Beacon__4_Colors_3b = $4E73
!Beacon__4_Colors_4b = $4631
!Beacon__4_Colors_5b = $3DEF

!Beacon__5_Colors_0a = $02BF, $017F, $0015
!Beacon__5_Colors_1a = $023B, $00FB, $0011
!Beacon__5_Colors_2a = $01D8, $0098, $000E
!Beacon__5_Colors_3a = $0154, $0055, $000B
!Beacon__5_Colors_4a = $00D0, $0010, $0007
!Beacon__5_Colors_5a = $00AA, $000B, $0004
!Beacon__5_Colors_0b = $7FFF
!Beacon__5_Colors_1b = $739C
!Beacon__5_Colors_2b = $5AD6
!Beacon__5_Colors_3b = $4E73
!Beacon__5_Colors_4b = $4631
!Beacon__5_Colors_5b = $3DEF

!Beacon__6_Colors_0a = $02BF, $017F, $0015
!Beacon__6_Colors_1a = $023B, $00FB, $0011
!Beacon__6_Colors_2a = $01D8, $0098, $000E
!Beacon__6_Colors_3a = $0154, $0055, $000B
!Beacon__6_Colors_4a = $00D0, $0010, $0007
!Beacon__6_Colors_5a = $00AA, $000B, $0004
!Beacon__6_Colors_0b = $7FFF
!Beacon__6_Colors_1b = $739C
!Beacon__6_Colors_2b = $5AD6
!Beacon__6_Colors_3b = $4E73
!Beacon__6_Colors_4b = $4631
!Beacon__6_Colors_5b = $3DEF

!Beacon__7_Colors_0a = $02BF, $017F, $0015
!Beacon__7_Colors_1a = $023B, $00FB, $0011
!Beacon__7_Colors_2a = $01D8, $0098, $000E
!Beacon__7_Colors_3a = $0154, $0055, $000B
!Beacon__7_Colors_4a = $00D0, $0010, $0007
!Beacon__7_Colors_5a = $00AA, $000B, $0004
!Beacon__7_Colors_0b = $7FFF
!Beacon__7_Colors_1b = $739C
!Beacon__7_Colors_2b = $5AD6
!Beacon__7_Colors_3b = $4E73
!Beacon__7_Colors_4b = $4631
!Beacon__7_Colors_5b = $3DEF

macro Beacon___List(n)
Beacon__<n>_List:
  DW SetColorIndex, $00C8
Beacon__<n>_List_Loop:
  DW $000A
    DW !Beacon__<n>_Colors_0a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_0b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_1a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_1b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_2a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_2b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_3a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_3b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_4a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_4b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_5a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_5b
    DW GlowYeild
  DW PlaySFX : DB $18
  DW $000A
    DW !Beacon__<n>_Colors_5a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_5b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_4a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_4b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_3a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_3b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_2a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_2b
    DW GlowYeild
  DW $000A
    DW !Beacon__<n>_Colors_1a
    DW SkipColors_9
    DW !Beacon__<n>_Colors_1b
    DW GlowYeild
  DW GlowJMP, Beacon__<n>_List_Loop
endmacro

%Beacon___List(1)
%Beacon___List(2)
%Beacon___List(3)
%Beacon___List(4)
%Beacon___List(5)
%Beacon___List(6)
%Beacon___List(7)

; Norfair tileset glows

SetHeatGlowSync:
  LDA $0000,Y
  AND #$00FF
  STA $1EED
  INY
  RTS

NorfairCommonColors_0:
  DW $09FD, $093B, $0459
NorfairCommonColors_1:
  DW $0E3D, $0D7C, $089A
NorfairCommonColors_2:
  DW $165E, $0DBC, $08FB
NorfairCommonColors_3:
  DW $1A9E, $11FD, $0D3C
NorfairCommonColors_4:
  DW $1EBE, $161D, $119C
NorfairCommonColors_5:
  DW $22FE, $1A5E, $15DD
NorfairCommonColors_6:
  DW $2B1F, $1A9E, $163E
NorfairCommonColors_7:
  DW $2F5F, $1EDF, $1A7F

NorfairCommon:
  LDA $0000,Y
  STA $7EC000,X
  INX
  INX
  LDA $0002,Y
  STA $7EC000,X
  INX
  INX
  LDA $0004,Y
  STA $7EC000,X
  INX
  INX
  PLY
  INY
  INY
  RTS

NorfairCommon_0:
  PHY
  LDY #NorfairCommonColors_0
  JMP NorfairCommon

NorfairCommon_1:
  PHY
  LDY #NorfairCommonColors_1
  JMP NorfairCommon

NorfairCommon_2:
  PHY
  LDY #NorfairCommonColors_2
  JMP NorfairCommon

NorfairCommon_3:
  PHY
  LDY #NorfairCommonColors_3
  JMP NorfairCommon

NorfairCommon_4:
  PHY
  LDY #NorfairCommonColors_4
  JMP NorfairCommon

NorfairCommon_5:
  PHY
  LDY #NorfairCommonColors_5
  JMP NorfairCommon

NorfairCommon_6:
  PHY
  LDY #NorfairCommonColors_6
  JMP NorfairCommon

NorfairCommon_7:
  PHY
  LDY #NorfairCommonColors_7
  JMP NorfairCommon

NorfairCommonColorsInit:
  PHX
  PHY
  LDA $1EBD,Y
  TAY
  LDX $0002,Y
  LDA NorfairCommonColors_0+0
  STA $7EC200,X
  LDA NorfairCommonColors_0+2
  STA $7EC202,X
  LDA NorfairCommonColors_0+4
  STA $7EC204,X

  LDA #$0EDF
  STA $7EC268
  STA $7EC250
  LDA #$0E3F
  STA $7EC252
  LDA #$0D7F
  STA $7EC254
  LDA #$0C9F
  STA $7EC256
  LDA #$0EDF
  STA $7EC25A

  PLY
  PLX
  RTS

NorHot1_Table:
  DW NorfairCommonColorsInit,NorHot1_0_List, NorfairCommonColorsInit,NorHot1_1_List, NorfairCommonColorsInit,NorHot1_2_List, NorfairCommonColorsInit,NorHot1_3_List
  DW NorfairCommonColorsInit,NorHot1_4_List, NorfairCommonColorsInit,NorHot1_5_List, NorfairCommonColorsInit,NorHot1_6_List, NorfairCommonColorsInit,NorHot1_7_List

!NorHot1_0_Colors_0 = $0E3F, $4A52
!NorHot1_0_Colors_1 = $125F, $4234
!NorHot1_0_Colors_2 = $169F, $3A16
!NorHot1_0_Colors_3 = $1ABF, $31F8
!NorHot1_0_Colors_4 = $22DF, $25D9
!NorHot1_0_Colors_5 = $26FF, $1DBB
!NorHot1_0_Colors_6 = $2B3F, $159D
!NorHot1_0_Colors_7 = $2F5F, $0D7F

!NorHot1_1_Colors_0 = $09FD, $4A52
!NorHot1_1_Colors_1 = $0E3D, $4214
!NorHot1_1_Colors_2 = $165E, $39F5
!NorHot1_1_Colors_3 = $1A9E, $31D7
!NorHot1_1_Colors_4 = $1EBE, $29D9
!NorHot1_1_Colors_5 = $22FE, $21BA
!NorHot1_1_Colors_6 = $2B1F, $199C
!NorHot1_1_Colors_7 = $2F5F, $0D7F

!NorHot1_2_Colors_0 = $09FD, $4A52
!NorHot1_2_Colors_1 = $0E3D, $4214
!NorHot1_2_Colors_2 = $165E, $39F5
!NorHot1_2_Colors_3 = $1A9E, $31D7
!NorHot1_2_Colors_4 = $1EBE, $29D9
!NorHot1_2_Colors_5 = $22FE, $21BA
!NorHot1_2_Colors_6 = $2B1F, $199C
!NorHot1_2_Colors_7 = $2F5F, $0D7F

!NorHot1_3_Colors_0 = $09FD, $4A52
!NorHot1_3_Colors_1 = $0E3D, $4214
!NorHot1_3_Colors_2 = $165E, $39F5
!NorHot1_3_Colors_3 = $1A9E, $31D7
!NorHot1_3_Colors_4 = $1EBE, $29D9
!NorHot1_3_Colors_5 = $22FE, $21BA
!NorHot1_3_Colors_6 = $2B1F, $199C
!NorHot1_3_Colors_7 = $2F5F, $0D7F

!NorHot1_4_Colors_0 = $09FD, $4A52
!NorHot1_4_Colors_1 = $0E3D, $4214
!NorHot1_4_Colors_2 = $165E, $39F5
!NorHot1_4_Colors_3 = $1A9E, $31D7
!NorHot1_4_Colors_4 = $1EBE, $29D9
!NorHot1_4_Colors_5 = $22FE, $21BA
!NorHot1_4_Colors_6 = $2B1F, $199C
!NorHot1_4_Colors_7 = $2F5F, $0D7F

!NorHot1_5_Colors_0 = $09FD, $4A52
!NorHot1_5_Colors_1 = $0E3D, $4214
!NorHot1_5_Colors_2 = $165E, $39F5
!NorHot1_5_Colors_3 = $1A9E, $31D7
!NorHot1_5_Colors_4 = $1EBE, $29D9
!NorHot1_5_Colors_5 = $22FE, $21BA
!NorHot1_5_Colors_6 = $2B1F, $199C
!NorHot1_5_Colors_7 = $2F5F, $0D7F

!NorHot1_6_Colors_0 = $09FD, $4A52
!NorHot1_6_Colors_1 = $0E3D, $4214
!NorHot1_6_Colors_2 = $165E, $39F5
!NorHot1_6_Colors_3 = $1A9E, $31D7
!NorHot1_6_Colors_4 = $1EBE, $29D9
!NorHot1_6_Colors_5 = $22FE, $21BA
!NorHot1_6_Colors_6 = $2B1F, $199C
!NorHot1_6_Colors_7 = $2F5F, $0D7F

!NorHot1_7_Colors_0 = $09FD, $4A52
!NorHot1_7_Colors_1 = $0E3D, $4214
!NorHot1_7_Colors_2 = $165E, $39F5
!NorHot1_7_Colors_3 = $1A9E, $31D7
!NorHot1_7_Colors_4 = $1EBE, $29D9
!NorHot1_7_Colors_5 = $22FE, $21BA
!NorHot1_7_Colors_6 = $2B1F, $199C
!NorHot1_7_Colors_7 = $2F5F, $0D7F

macro NorHot1__List(n)
NorHot1_<n>_List:
  DW SetColorIndex, $006A
NorHot1_<n>_List_Loop:
  DW SetHeatGlowSync : DB $00
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_0
    DW GlowYeild
  DW SetHeatGlowSync : DB $01
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_1
    DW GlowYeild
  DW SetHeatGlowSync : DB $02
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_2
    DW GlowYeild
  DW SetHeatGlowSync : DB $03
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_3
    DW GlowYeild
  DW SetHeatGlowSync : DB $04
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_4
    DW GlowYeild
  DW SetHeatGlowSync : DB $05
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_5
    DW GlowYeild
  DW SetHeatGlowSync : DB $06
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_6
    DW GlowYeild
  DW SetHeatGlowSync : DB $07
  DW $0008
    DW NorfairCommon_7
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_7
    DW GlowYeild
  DW SetHeatGlowSync : DB $08
  DW $0008
    DW NorfairCommon_7
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_7
    DW GlowYeild
  DW SetHeatGlowSync : DB $09
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_6
    DW GlowYeild
  DW SetHeatGlowSync : DB $0A
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_5
    DW GlowYeild
  DW SetHeatGlowSync : DB $0B
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_4
    DW GlowYeild
  DW SetHeatGlowSync : DB $0C
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_3
    DW GlowYeild
  DW SetHeatGlowSync : DB $0D
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_2
    DW GlowYeild
  DW SetHeatGlowSync : DB $0E
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_1
    DW GlowYeild
  DW SetHeatGlowSync : DB $0F
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_4
    DW !NorHot1_<n>_Colors_0
    DW GlowYeild
  DW GlowJMP, NorHot1_<n>_List_Loop
endmacro

%NorHot1__List(0)
%NorHot1__List(1)
%NorHot1__List(2)
%NorHot1__List(3)
%NorHot1__List(4)
%NorHot1__List(5)
%NorHot1__List(6)
%NorHot1__List(7)

NorHot2_Table:
  DW NorfairCommonColorsInit,NorHot2_0_List, NorfairCommonColorsInit,NorHot2_1_List, NorfairCommonColorsInit,NorHot2_2_List, NorfairCommonColorsInit,NorHot2_3_List
  DW NorfairCommonColorsInit,NorHot2_4_List, NorfairCommonColorsInit,NorHot2_5_List, NorfairCommonColorsInit,NorHot2_6_List, NorfairCommonColorsInit,NorHot2_7_List

!NorHot2_0_Colors_0 = $2C06, $2108
!NorHot2_0_Colors_1 = $284A, $214B
!NorHot2_0_Colors_2 = $246D, $210F
!NorHot2_0_Colors_3 = $2031, $2152
!NorHot2_0_Colors_4 = $1854, $2235
!NorHot2_0_Colors_5 = $1518, $2278
!NorHot2_0_Colors_6 = $113B, $223C
!NorHot2_0_Colors_7 = $0D7F, $227F

!NorHot2_1_Colors_0 = $4309, $0C77
!NorHot2_1_Colors_1 = $36AC, $0CB8
!NorHot2_1_Colors_2 = $328F, $1119
!NorHot2_1_Colors_3 = $2A52, $157A
!NorHot2_1_Colors_4 = $2214, $15BB
!NorHot2_1_Colors_5 = $1DF7, $1A1C
!NorHot2_1_Colors_6 = $15BA, $1E7D
!NorHot2_1_Colors_7 = $0D7F, $22FF

!NorHot2_2_Colors_0 = $4309, $0C77
!NorHot2_2_Colors_1 = $36AC, $0CB8
!NorHot2_2_Colors_2 = $328F, $1119
!NorHot2_2_Colors_3 = $2A52, $157A
!NorHot2_2_Colors_4 = $2214, $15BB
!NorHot2_2_Colors_5 = $1DF7, $1A1C
!NorHot2_2_Colors_6 = $15BA, $1E7D
!NorHot2_2_Colors_7 = $0D7F, $22FF

!NorHot2_3_Colors_0 = $4309, $0C77
!NorHot2_3_Colors_1 = $36AC, $0CB8
!NorHot2_3_Colors_2 = $328F, $1119
!NorHot2_3_Colors_3 = $2A52, $157A
!NorHot2_3_Colors_4 = $2214, $15BB
!NorHot2_3_Colors_5 = $1DF7, $1A1C
!NorHot2_3_Colors_6 = $15BA, $1E7D
!NorHot2_3_Colors_7 = $0D7F, $22FF

!NorHot2_4_Colors_0 = $4309, $0C77
!NorHot2_4_Colors_1 = $36AC, $0CB8
!NorHot2_4_Colors_2 = $328F, $1119
!NorHot2_4_Colors_3 = $2A52, $157A
!NorHot2_4_Colors_4 = $2214, $15BB
!NorHot2_4_Colors_5 = $1DF7, $1A1C
!NorHot2_4_Colors_6 = $15BA, $1E7D
!NorHot2_4_Colors_7 = $0D7F, $22FF

!NorHot2_5_Colors_0 = $4309, $0C77
!NorHot2_5_Colors_1 = $36AC, $0CB8
!NorHot2_5_Colors_2 = $328F, $1119
!NorHot2_5_Colors_3 = $2A52, $157A
!NorHot2_5_Colors_4 = $2214, $15BB
!NorHot2_5_Colors_5 = $1DF7, $1A1C
!NorHot2_5_Colors_6 = $15BA, $1E7D
!NorHot2_5_Colors_7 = $0D7F, $22FF

!NorHot2_6_Colors_0 = $4309, $0C77
!NorHot2_6_Colors_1 = $36AC, $0CB8
!NorHot2_6_Colors_2 = $328F, $1119
!NorHot2_6_Colors_3 = $2A52, $157A
!NorHot2_6_Colors_4 = $2214, $15BB
!NorHot2_6_Colors_5 = $1DF7, $1A1C
!NorHot2_6_Colors_6 = $15BA, $1E7D
!NorHot2_6_Colors_7 = $0D7F, $22FF

!NorHot2_7_Colors_0 = $4309, $0C77
!NorHot2_7_Colors_1 = $36AC, $0CB8
!NorHot2_7_Colors_2 = $328F, $1119
!NorHot2_7_Colors_3 = $2A52, $157A
!NorHot2_7_Colors_4 = $2214, $15BB
!NorHot2_7_Colors_5 = $1DF7, $1A1C
!NorHot2_7_Colors_6 = $15BA, $1E7D
!NorHot2_7_Colors_7 = $0D7F, $22FF

macro NorHot2__List(n)
NorHot2_<n>_List:
  DW SetColorIndex, $0082
NorHot2_<n>_List_Loop:
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_0
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_1
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_2
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_3
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_4
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_5
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_6
    DW GlowYeild
  DW $0010
    DW NorfairCommon_7
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_7
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_6
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_5
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_4
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_3
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_2
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_1
    DW GlowYeild
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot2_<n>_Colors_0
    DW GlowYeild
  DW GlowJMP, NorHot2_<n>_List_Loop
endmacro

%NorHot2__List(0)
%NorHot2__List(1)
%NorHot2__List(2)
%NorHot2__List(3)
%NorHot2__List(4)
%NorHot2__List(5)
%NorHot2__List(6)
%NorHot2__List(7)

NorHot3_Table:
  DW NorfairCommonColorsInit,NorHot3_0_List, NorfairCommonColorsInit,NorHot3_1_List, NorfairCommonColorsInit,NorHot3_2_List, NorfairCommonColorsInit,NorHot3_3_List
  DW NorfairCommonColorsInit,NorHot3_4_List, NorfairCommonColorsInit,NorHot3_5_List, NorfairCommonColorsInit,NorHot3_6_List, NorfairCommonColorsInit,NorHot3_7_List

!NorHot3_0_Colors_0 = $3DB3, $1404
!NorHot3_0_Colors_1 = $3595, $1428
!NorHot3_0_Colors_2 = $3196, $104C
!NorHot3_0_Colors_3 = $2978, $1070
!NorHot3_0_Colors_4 = $215A, $10B3
!NorHot3_0_Colors_5 = $193C, $10D7
!NorHot3_0_Colors_6 = $153D, $0CFB
!NorHot3_0_Colors_7 = $0D1F, $0D1F

!NorHot3_1_Colors_0 = $2DB3, $38CF
!NorHot3_1_Colors_1 = $2594, $30D1
!NorHot3_1_Colors_2 = $2176, $28D3
!NorHot3_1_Colors_3 = $1D57, $24D5
!NorHot3_1_Colors_4 = $1959, $20F7
!NorHot3_1_Colors_5 = $153B, $18F9
!NorHot3_1_Colors_6 = $111C, $14FB
!NorHot3_1_Colors_7 = $0D1F, $0D1F

!NorHot3_2_Colors_0 = $2DB3, $38CF
!NorHot3_2_Colors_1 = $2594, $30D1
!NorHot3_2_Colors_2 = $2176, $28D3
!NorHot3_2_Colors_3 = $1D57, $24D5
!NorHot3_2_Colors_4 = $1959, $20F7
!NorHot3_2_Colors_5 = $153B, $18F9
!NorHot3_2_Colors_6 = $111C, $14FB
!NorHot3_2_Colors_7 = $0D1F, $0D1F

!NorHot3_3_Colors_0 = $2DB3, $38CF
!NorHot3_3_Colors_1 = $2594, $30D1
!NorHot3_3_Colors_2 = $2176, $28D3
!NorHot3_3_Colors_3 = $1D57, $24D5
!NorHot3_3_Colors_4 = $1959, $20F7
!NorHot3_3_Colors_5 = $153B, $18F9
!NorHot3_3_Colors_6 = $111C, $14FB
!NorHot3_3_Colors_7 = $0D1F, $0D1F

!NorHot3_4_Colors_0 = $2DB3, $38CF
!NorHot3_4_Colors_1 = $2594, $30D1
!NorHot3_4_Colors_2 = $2176, $28D3
!NorHot3_4_Colors_3 = $1D57, $24D5
!NorHot3_4_Colors_4 = $1959, $20F7
!NorHot3_4_Colors_5 = $153B, $18F9
!NorHot3_4_Colors_6 = $111C, $14FB
!NorHot3_4_Colors_7 = $0D1F, $0D1F

!NorHot3_5_Colors_0 = $2DB3, $38CF
!NorHot3_5_Colors_1 = $2594, $30D1
!NorHot3_5_Colors_2 = $2176, $28D3
!NorHot3_5_Colors_3 = $1D57, $24D5
!NorHot3_5_Colors_4 = $1959, $20F7
!NorHot3_5_Colors_5 = $153B, $18F9
!NorHot3_5_Colors_6 = $111C, $14FB
!NorHot3_5_Colors_7 = $0D1F, $0D1F

!NorHot3_6_Colors_0 = $2DB3, $38CF
!NorHot3_6_Colors_1 = $2594, $30D1
!NorHot3_6_Colors_2 = $2176, $28D3
!NorHot3_6_Colors_3 = $1D57, $24D5
!NorHot3_6_Colors_4 = $1959, $20F7
!NorHot3_6_Colors_5 = $153B, $18F9
!NorHot3_6_Colors_6 = $111C, $14FB
!NorHot3_6_Colors_7 = $0D1F, $0D1F

!NorHot3_7_Colors_0 = $2DB3, $38CF
!NorHot3_7_Colors_1 = $2594, $30D1
!NorHot3_7_Colors_2 = $2176, $28D3
!NorHot3_7_Colors_3 = $1D57, $24D5
!NorHot3_7_Colors_4 = $1959, $20F7
!NorHot3_7_Colors_5 = $153B, $18F9
!NorHot3_7_Colors_6 = $111C, $14FB
!NorHot3_7_Colors_7 = $0D1F, $0D1F

macro NorHot3__List(n)
NorHot3_<n>_List:
  DW SetColorIndex, $00A2
NorHot3_<n>_List_Loop:
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_0
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_1
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_2
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_3
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_4
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_5
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_6
    DW GlowYeild
  DW $0010
    DW NorfairCommon_7
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_7
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_6
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_5
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_4
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_3
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_2
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_1
    DW GlowYeild
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot3_<n>_Colors_0
    DW GlowYeild
  DW GlowJMP, NorHot3_<n>_List_Loop
endmacro

%NorHot3__List(0)
%NorHot3__List(1)
%NorHot3__List(2)
%NorHot3__List(3)
%NorHot3__List(4)
%NorHot3__List(5)
%NorHot3__List(6)
%NorHot3__List(7)

NorfairCommonDark_0:
  DW $09DA, $091A, $087A
NorfairCommonDark_1:
  DW $0DDA, $093A, $089A
NorfairCommonDark_2:
  DW $0DFA, $0D5A, $08BA
NorfairCommonDark_3:
  DW $11FA, $0D7A, $08FA
NorfairCommonDark_4:
  DW $161A, $119A, $0D1A
NorfairCommonDark_5:
  DW $1A1A, $11BA, $0D3A
NorfairCommonDark_6:
  DW $1A3A, $15DA, $0D7A
NorfairCommonDark_7:
  DW $225A, $1A1A, $11BA

NorfairDark_0:
  PHY
  LDY #NorfairCommonDark_0
  JMP NorfairCommon

NorfairDark_1:
  PHY
  LDY #NorfairCommonDark_1
  JMP NorfairCommon

NorfairDark_2:
  PHY
  LDY #NorfairCommonDark_2
  JMP NorfairCommon

NorfairDark_3:
  PHY
  LDY #NorfairCommonDark_3
  JMP NorfairCommon

NorfairDark_4:
  PHY
  LDY #NorfairCommonDark_4
  JMP NorfairCommon

NorfairDark_5:
  PHY
  LDY #NorfairCommonDark_5
  JMP NorfairCommon

NorfairDark_6:
  PHY
  LDY #NorfairCommonDark_6
  JMP NorfairCommon

NorfairDark_7:
  PHY
  LDY #NorfairCommonDark_7
  JMP NorfairCommon

NorfairCommonDarkInit:
  PHX
  PHY
  LDA $1EBD,Y
  TAY
  LDX $0002,Y
  LDA NorfairCommonDark_0+0
  STA $7EC200,X
  LDA NorfairCommonDark_0+2
  STA $7EC202,X
  LDA NorfairCommonDark_0+4
  STA $7EC204,X

  LDA #$0596
  STA $7EC220,X
  LDA #$04D6
  STA $7EC222,X
  LDA #$0456
  STA $7EC224,X

  PLY
  PLX
  RTS

NorHot4_Table:
  DW NorfairCommonDarkInit,NorHot4_0_List, NorfairCommonDarkInit,NorHot4_1_List, NorfairCommonDarkInit,NorHot4_2_List, NorfairCommonDarkInit,NorHot4_3_List
  DW NorfairCommonDarkInit,NorHot4_4_List, NorfairCommonDarkInit,NorHot4_5_List, NorfairCommonDarkInit,NorHot4_6_List, NorfairCommonDarkInit,NorHot4_7_List

!NorHot4_0_Colors_0 = $1CA6, $0C43
!NorHot4_0_Colors_1 = $18A9, $0C66
!NorHot4_0_Colors_2 = $18CC, $0C8A
!NorHot4_0_Colors_3 = $14CF, $0CAD
!NorHot4_0_Colors_4 = $10F1, $08B0
!NorHot4_0_Colors_5 = $0CF4, $08D3
!NorHot4_0_Colors_6 = $0D17, $08F7
!NorHot4_0_Colors_7 = $091A, $091A

!NorHot4_1_Colors_0 = $08A8, $0C05
!NorHot4_1_Colors_1 = $08AA, $0828
!NorHot4_1_Colors_2 = $08AC, $084A
!NorHot4_1_Colors_3 = $08CF, $086D
!NorHot4_1_Colors_4 = $08D1, $0890
!NorHot4_1_Colors_5 = $08F4, $08B3
!NorHot4_1_Colors_6 = $08F6, $08D5
!NorHot4_1_Colors_7 = $091A, $091A

!NorHot4_2_Colors_0 = $08A8, $0C05
!NorHot4_2_Colors_1 = $08AA, $0828
!NorHot4_2_Colors_2 = $08AC, $084A
!NorHot4_2_Colors_3 = $08CF, $086D
!NorHot4_2_Colors_4 = $08D1, $0890
!NorHot4_2_Colors_5 = $08F4, $08B3
!NorHot4_2_Colors_6 = $08F6, $08D5
!NorHot4_2_Colors_7 = $091A, $091A

!NorHot4_3_Colors_0 = $08A8, $0C05
!NorHot4_3_Colors_1 = $08AA, $0828
!NorHot4_3_Colors_2 = $08AC, $084A
!NorHot4_3_Colors_3 = $08CF, $086D
!NorHot4_3_Colors_4 = $08D1, $0890
!NorHot4_3_Colors_5 = $08F4, $08B3
!NorHot4_3_Colors_6 = $08F6, $08D5
!NorHot4_3_Colors_7 = $091A, $091A

!NorHot4_4_Colors_0 = $08A8, $0C05
!NorHot4_4_Colors_1 = $08AA, $0828
!NorHot4_4_Colors_2 = $08AC, $084A
!NorHot4_4_Colors_3 = $08CF, $086D
!NorHot4_4_Colors_4 = $08D1, $0890
!NorHot4_4_Colors_5 = $08F4, $08B3
!NorHot4_4_Colors_6 = $08F6, $08D5
!NorHot4_4_Colors_7 = $091A, $091A

!NorHot4_5_Colors_0 = $08A8, $0C05
!NorHot4_5_Colors_1 = $08AA, $0828
!NorHot4_5_Colors_2 = $08AC, $084A
!NorHot4_5_Colors_3 = $08CF, $086D
!NorHot4_5_Colors_4 = $08D1, $0890
!NorHot4_5_Colors_5 = $08F4, $08B3
!NorHot4_5_Colors_6 = $08F6, $08D5
!NorHot4_5_Colors_7 = $091A, $091A

!NorHot4_6_Colors_0 = $08A8, $0C05
!NorHot4_6_Colors_1 = $08AA, $0828
!NorHot4_6_Colors_2 = $08AC, $084A
!NorHot4_6_Colors_3 = $08CF, $086D
!NorHot4_6_Colors_4 = $08D1, $0890
!NorHot4_6_Colors_5 = $08F4, $08B3
!NorHot4_6_Colors_6 = $08F6, $08D5
!NorHot4_6_Colors_7 = $091A, $091A

!NorHot4_7_Colors_0 = $08A8, $0C05
!NorHot4_7_Colors_1 = $08AA, $0828
!NorHot4_7_Colors_2 = $08AC, $084A
!NorHot4_7_Colors_3 = $08CF, $086D
!NorHot4_7_Colors_4 = $08D1, $0890
!NorHot4_7_Colors_5 = $08F4, $08B3
!NorHot4_7_Colors_6 = $08F6, $08D5
!NorHot4_7_Colors_7 = $091A, $091A

macro NorHot4__List(n)
NorHot4_<n>_List:
  DW SetColorIndex, $00C2
NorHot4_<n>_List_Loop:
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_0
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_1
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_2
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_3
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_4
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_5
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_6
    DW GlowYeild
  DW $0010
    DW NorfairCommon_7
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_7
    DW GlowYeild
  DW $0008
    DW NorfairCommon_6
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_6
    DW GlowYeild
  DW $0007
    DW NorfairCommon_5
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_5
    DW GlowYeild
  DW $0006
    DW NorfairCommon_4
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_4
    DW GlowYeild
  DW $0005
    DW NorfairCommon_3
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_3
    DW GlowYeild
  DW $0004
    DW NorfairCommon_2
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_2
    DW GlowYeild
  DW $0004
    DW NorfairCommon_1
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_1
    DW GlowYeild
  DW $0010
    DW NorfairCommon_0
    DW SkipColors_8
    DW !NorHot4_<n>_Colors_0
    DW GlowYeild
  DW GlowJMP, NorHot4_<n>_List_Loop
endmacro

%NorHot4__List(0)
%NorHot4__List(1)
%NorHot4__List(2)
%NorHot4__List(3)
%NorHot4__List(4)
%NorHot4__List(5)
%NorHot4__List(6)
%NorHot4__List(7)

; Wrecked Ship tileset glows

WS_GreenTable:
  DW EmptyInit,WS_Green0_List, EmptyInit,WS_Green1_List, EmptyInit,WS_Green2_List, EmptyInit,WS_Green3_List
  DW EmptyInit,WS_Green4_List, EmptyInit,WS_Green5_List, EmptyInit,WS_Green6_List, EmptyInit,WS_Green7_List

!WS_Green0_Colors_0 = $2C06, $5C69
!WS_Green0_Colors_1 = $3408, $64EC
!WS_Green0_Colors_2 = $4009, $6D4F
!WS_Green0_Colors_3 = $4C0A, $71B2
!WS_Green0_Colors_4 = $540C, $7A35

!WS_Green1_Colors_0 = $1EA9, $0BB1
!WS_Green1_Colors_1 = $1667, $034E
!WS_Green1_Colors_2 = $0E25, $02EB
!WS_Green1_Colors_3 = $05E3, $0288
!WS_Green1_Colors_4 = $01A1, $0225

!WS_Green2_Colors_0 = $1EA9, $0BB1
!WS_Green2_Colors_1 = $1667, $034E
!WS_Green2_Colors_2 = $0E25, $02EB
!WS_Green2_Colors_3 = $05E3, $0288
!WS_Green2_Colors_4 = $01A1, $0225

!WS_Green3_Colors_0 = $1EA9, $0BB1
!WS_Green3_Colors_1 = $1667, $034E
!WS_Green3_Colors_2 = $0E25, $02EB
!WS_Green3_Colors_3 = $05E3, $0288
!WS_Green3_Colors_4 = $01A1, $0225

!WS_Green4_Colors_0 = $1EA9, $0BB1
!WS_Green4_Colors_1 = $1667, $034E
!WS_Green4_Colors_2 = $0E25, $02EB
!WS_Green4_Colors_3 = $05E3, $0288
!WS_Green4_Colors_4 = $01A1, $0225

!WS_Green5_Colors_0 = $1EA9, $0BB1
!WS_Green5_Colors_1 = $1667, $034E
!WS_Green5_Colors_2 = $0E25, $02EB
!WS_Green5_Colors_3 = $05E3, $0288
!WS_Green5_Colors_4 = $01A1, $0225

!WS_Green6_Colors_0 = $1EA9, $0BB1
!WS_Green6_Colors_1 = $1667, $034E
!WS_Green6_Colors_2 = $0E25, $02EB
!WS_Green6_Colors_3 = $05E3, $0288
!WS_Green6_Colors_4 = $01A1, $0225

!WS_Green7_Colors_0 = $1EA9, $0BB1
!WS_Green7_Colors_1 = $1667, $034E
!WS_Green7_Colors_2 = $0E25, $02EB
!WS_Green7_Colors_3 = $05E3, $0288
!WS_Green7_Colors_4 = $01A1, $0225

macro WS_Green_List(n)
WS_Green<n>_List:
  DW SetColorIndex, $0098
WS_Green<n>_List_Loop:
  DW $000A
    DW !WS_Green<n>_Colors_0
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_1
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_4
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_3
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_2
    DW GlowYeild
  DW $000A
    DW !WS_Green<n>_Colors_1
    DW GlowYeild
  DW GlowJMP, WS_Green<n>_List_Loop
endmacro

%WS_Green_List(0)
%WS_Green_List(1)
%WS_Green_List(2)
%WS_Green_List(3)
%WS_Green_List(4)
%WS_Green_List(5)
%WS_Green_List(6)
%WS_Green_List(7)

; Maridia tileset glows

WaterfallColorsInit:
  PHX
  PHY
  LDA $1EBD,Y
  CLC
  ADC #$0006
  STA $12

  LDY #$0000
  LDX #$0068
-
  LDA ($12),Y
  STA $7EC200,X
  INX
  INX
  INY
  INY
  CPY #$0010
  BMI -
  PLY
  PLX
  RTS

WaterfalTable:
  DW WaterfallColorsInit,Waterfal0_List, WaterfallColorsInit,Waterfal1_List, WaterfallColorsInit,Waterfal2_List, WaterfallColorsInit,Waterfal3_List
  DW WaterfallColorsInit,Waterfal4_List, WaterfallColorsInit,Waterfal5_List, WaterfallColorsInit,Waterfal6_List, WaterfallColorsInit,Waterfal7_List

!Waterfal0_Color_0 = $0400
!Waterfal0_Color_7 = $0821
!Waterfal0_Color_1 = $0C22
!Waterfal0_Color_6 = $1043
!Waterfal0_Color_2 = $1864
!Waterfal0_Color_5 = $1C65
!Waterfal0_Color_3 = $2086
!Waterfal0_Color_4 = $2CC9

!Waterfal1_Color_0 = $0400
!Waterfal1_Color_7 = $0821
!Waterfal1_Color_1 = $0C22
!Waterfal1_Color_6 = $1043
!Waterfal1_Color_2 = $1864
!Waterfal1_Color_5 = $1C65
!Waterfal1_Color_3 = $2086
!Waterfal1_Color_4 = $2CC9

!Waterfal2_Color_0 = $0400
!Waterfal2_Color_7 = $0821
!Waterfal2_Color_1 = $0C22
!Waterfal2_Color_6 = $1043
!Waterfal2_Color_2 = $1864
!Waterfal2_Color_5 = $1C65
!Waterfal2_Color_3 = $2086
!Waterfal2_Color_4 = $2CC9

!Waterfal3_Color_0 = $0400
!Waterfal3_Color_7 = $0821
!Waterfal3_Color_1 = $0C22
!Waterfal3_Color_6 = $1043
!Waterfal3_Color_2 = $1864
!Waterfal3_Color_5 = $1C65
!Waterfal3_Color_3 = $2086
!Waterfal3_Color_4 = $2CC9

!Waterfal4_Color_0 = $0400
!Waterfal4_Color_7 = $0821
!Waterfal4_Color_1 = $0C22
!Waterfal4_Color_6 = $1043
!Waterfal4_Color_2 = $1864
!Waterfal4_Color_5 = $1C65
!Waterfal4_Color_3 = $2086
!Waterfal4_Color_4 = $2CC9

!Waterfal5_Color_0 = $0400
!Waterfal5_Color_7 = $0821
!Waterfal5_Color_1 = $0C22
!Waterfal5_Color_6 = $1043
!Waterfal5_Color_2 = $1864
!Waterfal5_Color_5 = $1C65
!Waterfal5_Color_3 = $2086
!Waterfal5_Color_4 = $2CC9

!Waterfal6_Color_0 = $0400
!Waterfal6_Color_7 = $0821
!Waterfal6_Color_1 = $0C22
!Waterfal6_Color_6 = $1043
!Waterfal6_Color_2 = $1864
!Waterfal6_Color_5 = $1C65
!Waterfal6_Color_3 = $2086
!Waterfal6_Color_4 = $2CC9

!Waterfal7_Color_0 = $0400
!Waterfal7_Color_7 = $0821
!Waterfal7_Color_1 = $0C22
!Waterfal7_Color_6 = $1043
!Waterfal7_Color_2 = $1864
!Waterfal7_Color_5 = $1C65
!Waterfal7_Color_3 = $2086
!Waterfal7_Color_4 = $2CC9

macro Waterfal_List(n)
Waterfal<n>_List:
  DW SetColorIndex, $0068
Waterfal<n>_List_Loop:
  DW $0002
    DW !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_5, !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_6, !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5
    DW GlowYeild
  DW $0002
    DW !Waterfal<n>_Color_7, !Waterfal<n>_Color_0, !Waterfal<n>_Color_1, !Waterfal<n>_Color_2, !Waterfal<n>_Color_3, !Waterfal<n>_Color_4, !Waterfal<n>_Color_5, !Waterfal<n>_Color_6
    DW GlowYeild
  DW GlowJMP, Waterfal<n>_List_Loop
endmacro

%Waterfal_List(0)
%Waterfal_List(1)
%Waterfal_List(2)
%Waterfal_List(3)
%Waterfal_List(4)
%Waterfal_List(5)
%Waterfal_List(6)
%Waterfal_List(7)

; Tourian tileset glows
Tourian_PreInstruction:
  LDA $1E79,X
  BEQ +
  STZ $1E7D,X
+
  RTS

Tourian_Table:
  DW EmptyInit,Tourian_0_List, EmptyInit,Tourian_1_List, EmptyInit,Tourian_2_List, EmptyInit,Tourian_3_List
  DW EmptyInit,Tourian_4_List, EmptyInit,Tourian_5_List, EmptyInit,Tourian_6_List, EmptyInit,Tourian_7_List

!Tourian_0_Colors_0a = $5A73
!Tourian_0_Colors_1a = $4E10
!Tourian_0_Colors_2a = $3DAD
!Tourian_0_Colors_3a = $3129
!Tourian_0_Colors_4a = $20C6
!Tourian_0_Colors_5a = $1463
!Tourian_0_Colors_0b = $3412, $240B, $3460, $1840, $1084, $517F, $7FFF
!Tourian_0_Colors_1b = $3011, $200A, $3040, $1840, $1084, $451C, $739C
!Tourian_0_Colors_2b = $2C0F, $200A, $2C40, $1420, $1084, $38B9, $6739
!Tourian_0_Colors_3b = $2C0E, $1C09, $2820, $1420, $1084, $4456, $5AD6
!Tourian_0_Colors_4b = $280C, $1C09, $2420, $1000, $1084, $4413, $4E73
!Tourian_0_Colors_5b = $240B, $1808, $2000, $0400, $1084, $3410, $4210

!Tourian_1_Colors_0a = $5294
!Tourian_1_Colors_1a = $4A52
!Tourian_1_Colors_2a = $4210
!Tourian_1_Colors_3a = $39CE
!Tourian_1_Colors_4a = $318C
!Tourian_1_Colors_5a = $294A
!Tourian_1_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_1_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_1_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_1_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_1_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_1_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_2_Colors_0a = $5294
!Tourian_2_Colors_1a = $4A52
!Tourian_2_Colors_2a = $4210
!Tourian_2_Colors_3a = $39CE
!Tourian_2_Colors_4a = $318C
!Tourian_2_Colors_5a = $294A
!Tourian_2_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_2_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_2_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_2_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_2_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_2_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_3_Colors_0a = $5294
!Tourian_3_Colors_1a = $4A52
!Tourian_3_Colors_2a = $4210
!Tourian_3_Colors_3a = $39CE
!Tourian_3_Colors_4a = $318C
!Tourian_3_Colors_5a = $294A
!Tourian_3_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_3_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_3_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_3_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_3_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_3_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_4_Colors_0a = $5294
!Tourian_4_Colors_1a = $4A52
!Tourian_4_Colors_2a = $4210
!Tourian_4_Colors_3a = $39CE
!Tourian_4_Colors_4a = $318C
!Tourian_4_Colors_5a = $294A
!Tourian_4_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_4_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_4_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_4_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_4_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_4_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_5_Colors_0a = $5294
!Tourian_5_Colors_1a = $4A52
!Tourian_5_Colors_2a = $4210
!Tourian_5_Colors_3a = $39CE
!Tourian_5_Colors_4a = $318C
!Tourian_5_Colors_5a = $294A
!Tourian_5_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_5_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_5_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_5_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_5_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_5_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_6_Colors_0a = $5294
!Tourian_6_Colors_1a = $4A52
!Tourian_6_Colors_2a = $4210
!Tourian_6_Colors_3a = $39CE
!Tourian_6_Colors_4a = $318C
!Tourian_6_Colors_5a = $294A
!Tourian_6_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_6_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_6_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_6_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_6_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_6_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

!Tourian_7_Colors_0a = $5294
!Tourian_7_Colors_1a = $4A52
!Tourian_7_Colors_2a = $4210
!Tourian_7_Colors_3a = $39CE
!Tourian_7_Colors_4a = $318C
!Tourian_7_Colors_5a = $294A
!Tourian_7_Colors_0b = $0019, $0012, $5C00, $4000, $1084, $197F, $7FFF
!Tourian_7_Colors_1b = $0016, $000F, $5000, $3400, $1084, $0D1C, $739C
!Tourian_7_Colors_2b = $0013, $000C, $4400, $2800, $1084, $00B9, $6739
!Tourian_7_Colors_3b = $0010, $0009, $3800, $1C00, $1084, $0056, $5AD6
!Tourian_7_Colors_4b = $000D, $0006, $2C00, $1000, $1084, $0013, $4E73
!Tourian_7_Colors_5b = $000A, $0003, $2000, $0400, $1084, $0010, $4210

macro Tourian__List(n)
Tourian_<n>_List:
  DW SetColorIndex, $00E8
Tourian_<n>_List_Loop:
  DW $000A
    DW !Tourian_<n>_Colors_0a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_0b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_1a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_1b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_2a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_2b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_3a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_3b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_4a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_4b
    DW GlowYeild
  DW $0014
    DW !Tourian_<n>_Colors_5a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_5b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_4a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_4b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_3a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_3b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_2a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_2b
    DW GlowYeild
  DW $000A
    DW !Tourian_<n>_Colors_1a
    DW SkipColors_3
    DW !Tourian_<n>_Colors_1b
    DW GlowYeild
  DW GlowJMP, Tourian_<n>_List_Loop
endmacro

%Tourian__List(0)
%Tourian__List(1)
%Tourian__List(2)
%Tourian__List(3)
%Tourian__List(4)
%Tourian__List(5)
%Tourian__List(6)
%Tourian__List(7)

Tor_2EscTable:
  DW EmptyInit,Tor_2Esc0_List, EmptyInit,Tor_2Esc1_List, EmptyInit,Tor_2Esc2_List, EmptyInit,Tor_2Esc3_List
  DW EmptyInit,Tor_2Esc4_List, EmptyInit,Tor_2Esc5_List, EmptyInit,Tor_2Esc6_List, EmptyInit,Tor_2Esc7_List

!Tor_2Esc0_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc0_Colors_1 = $1038, $0C30, $084A, $0823
!Tor_2Esc0_Colors_2 = $1475, $104F, $0C49, $0843
!Tor_2Esc0_Colors_3 = $1C93, $146D, $1068, $0C43
!Tor_2Esc0_Colors_4 = $20D1, $1C8C, $1468, $0C43
!Tor_2Esc0_Colors_5 = $28EF, $20AA, $1887, $1043
!Tor_2Esc0_Colors_6 = $2D2C, $24C9, $1C86, $1063
!Tor_2Esc0_Colors_7 = $354A, $28E7, $20A5, $1463

!Tor_2Esc1_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc1_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc1_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc1_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc1_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc1_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc1_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc1_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc2_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc2_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc2_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc2_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc2_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc2_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc2_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc2_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc3_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc3_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc3_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc3_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc3_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc3_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc3_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc3_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc4_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc4_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc4_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc4_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc4_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc4_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc4_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc4_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc5_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc5_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc5_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc5_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc5_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc5_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc5_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc5_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc6_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc6_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc6_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc6_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc6_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc6_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc6_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc6_Colors_7 = $20E5, $18A4, $1083, $0841

!Tor_2Esc7_Colors_0 = $081A, $0812, $042B, $0423
!Tor_2Esc7_Colors_1 = $0C37, $0C30, $042A, $0423
!Tor_2Esc7_Colors_2 = $1054, $0C2E, $0849, $0422
!Tor_2Esc7_Colors_3 = $1471, $104C, $0848, $0422
!Tor_2Esc7_Colors_4 = $148E, $106A, $0C66, $0842
!Tor_2Esc7_Colors_5 = $18AB, $1488, $0C65, $0842
!Tor_2Esc7_Colors_6 = $1CC8, $1486, $1084, $0841
!Tor_2Esc7_Colors_7 = $20E5, $18A4, $1083, $0841


macro Tor_2Esc_List(n)
Tor_2Esc<n>_List:
  DW SetColorIndex, $0070
Tor_2Esc<n>_List_Loop:
  DW $0004
    DW !Tor_2Esc0_Colors_0
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_1
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_2
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_3
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_4
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_5
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_6 
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_7
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_6 
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_5
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_4
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_3
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_2
    DW GlowYeild
  DW $0004
    DW !Tor_2Esc0_Colors_1
    DW GlowYeild
  DW GlowJMP, Tor_2Esc<n>_List_Loop
endmacro

%Tor_2Esc_List(0)
%Tor_2Esc_List(1)
%Tor_2Esc_List(2)
%Tor_2Esc_List(3)
%Tor_2Esc_List(4)
%Tor_2Esc_List(5)
%Tor_2Esc_List(6)
%Tor_2Esc_List(7)

Tor_3EscTable:
  DW EmptyInit,Tor_3Esc0_List, EmptyInit,Tor_3Esc1_List, EmptyInit,Tor_3Esc2_List, EmptyInit,Tor_3Esc3_List
  DW EmptyInit,Tor_3Esc4_List, EmptyInit,Tor_3Esc5_List, EmptyInit,Tor_3Esc6_List, EmptyInit,Tor_3Esc7_List
Tor_3Esc0_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc0_List_Loop
Tor_3Esc1_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc1_List_Loop
Tor_3Esc2_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc2_List_Loop
Tor_3Esc3_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc3_List_Loop
Tor_3Esc4_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc4_List_Loop
Tor_3Esc5_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc5_List_Loop
Tor_3Esc6_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc6_List_Loop
Tor_3Esc7_List:
  DW SetColorIndex, $00A8
  DW GlowJMP, Tor_4Esc0_List_Loop

Tor_4EscTable:
  DW EmptyInit,Tor_4Esc0_List, EmptyInit,Tor_4Esc1_List, EmptyInit,Tor_4Esc2_List, EmptyInit,Tor_4Esc3_List
  DW EmptyInit,Tor_4Esc4_List, EmptyInit,Tor_4Esc5_List, EmptyInit,Tor_4Esc6_List, EmptyInit,Tor_4Esc7_List

!Tor_4Esc0_Colors_0a = $5A73, $412D, $2867, $1463, $3412, $240B
!Tor_4Esc0_Colors_1a = $5654, $3D0E, $2468, $1063, $2C0F, $200A
!Tor_4Esc0_Colors_2a = $4E34, $390E, $2048, $1043, $280C, $1C09
!Tor_4Esc0_Colors_3a = $4A15, $356F, $1C49, $0C43, $2C0F, $200A
!Tor_4Esc0_Colors_4a = $4176, $2D50, $1C29, $0824, $3412, $240B
!Tor_4Esc0_Colors_5a = $3D57, $2931, $182A, $0424, $2C0F, $200A
!Tor_4Esc0_Colors_6a = $3537, $2531, $140A, $0404, $280C, $1C09
!Tor_4Esc0_Colors_7a = $3118, $2112, $100B, $0004, $2C0F, $200A
!Tor_4Esc0_Colors_0b = $7FFF
!Tor_4Esc0_Colors_1b = $77BF
!Tor_4Esc0_Colors_2b = $739F
!Tor_4Esc0_Colors_3b = $6B5F
!Tor_4Esc0_Colors_4b = $673F
!Tor_4Esc0_Colors_5b = $5EFF
!Tor_4Esc0_Colors_6b = $5ADF
!Tor_4Esc0_Colors_7b = $529F

!Tor_4Esc1_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc1_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc1_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc1_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc1_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc1_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc1_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc1_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc1_Colors_0b = $7FFF
!Tor_4Esc1_Colors_1b = $77BF
!Tor_4Esc1_Colors_2b = $739F
!Tor_4Esc1_Colors_3b = $6B5F
!Tor_4Esc1_Colors_4b = $673F
!Tor_4Esc1_Colors_5b = $5EFF
!Tor_4Esc1_Colors_6b = $5ADF
!Tor_4Esc1_Colors_7b = $529F

!Tor_4Esc2_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc2_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc2_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc2_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc2_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc2_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc2_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc2_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc2_Colors_0b = $7FFF
!Tor_4Esc2_Colors_1b = $77BF
!Tor_4Esc2_Colors_2b = $739F
!Tor_4Esc2_Colors_3b = $6B5F
!Tor_4Esc2_Colors_4b = $673F
!Tor_4Esc2_Colors_5b = $5EFF
!Tor_4Esc2_Colors_6b = $5ADF
!Tor_4Esc2_Colors_7b = $529F

!Tor_4Esc3_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc3_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc3_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc3_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc3_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc3_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc3_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc3_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc3_Colors_0b = $7FFF
!Tor_4Esc3_Colors_1b = $77BF
!Tor_4Esc3_Colors_2b = $739F
!Tor_4Esc3_Colors_3b = $6B5F
!Tor_4Esc3_Colors_4b = $673F
!Tor_4Esc3_Colors_5b = $5EFF
!Tor_4Esc3_Colors_6b = $5ADF
!Tor_4Esc3_Colors_7b = $529F

!Tor_4Esc4_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc4_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc4_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc4_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc4_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc4_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc4_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc4_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc4_Colors_0b = $7FFF
!Tor_4Esc4_Colors_1b = $77BF
!Tor_4Esc4_Colors_2b = $739F
!Tor_4Esc4_Colors_3b = $6B5F
!Tor_4Esc4_Colors_4b = $673F
!Tor_4Esc4_Colors_5b = $5EFF
!Tor_4Esc4_Colors_6b = $5ADF
!Tor_4Esc4_Colors_7b = $529F

!Tor_4Esc5_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc5_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc5_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc5_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc5_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc5_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc5_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc5_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc5_Colors_0b = $7FFF
!Tor_4Esc5_Colors_1b = $77BF
!Tor_4Esc5_Colors_2b = $739F
!Tor_4Esc5_Colors_3b = $6B5F
!Tor_4Esc5_Colors_4b = $673F
!Tor_4Esc5_Colors_5b = $5EFF
!Tor_4Esc5_Colors_6b = $5ADF
!Tor_4Esc5_Colors_7b = $529F

!Tor_4Esc6_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc6_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc6_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc6_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc6_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc6_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc6_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc6_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc6_Colors_0b = $7FFF
!Tor_4Esc6_Colors_1b = $77BF
!Tor_4Esc6_Colors_2b = $739F
!Tor_4Esc6_Colors_3b = $6B5F
!Tor_4Esc6_Colors_4b = $673F
!Tor_4Esc6_Colors_5b = $5EFF
!Tor_4Esc6_Colors_6b = $5ADF
!Tor_4Esc6_Colors_7b = $529F

!Tor_4Esc7_Colors_0a = $5294, $39CE, $2108, $1084, $0019, $0012
!Tor_4Esc7_Colors_1a = $4E75, $35AF, $1CE8, $0C64, $080D, $0809
!Tor_4Esc7_Colors_2a = $4A55, $318F, $1CE9, $0C64, $1000, $1000
!Tor_4Esc7_Colors_3a = $4636, $2D70, $18C9, $0844, $080D, $0809
!Tor_4Esc7_Colors_4a = $3DF6, $2D70, $18CA, $0844, $0019, $0012
!Tor_4Esc7_Colors_5a = $39D7, $2951, $14AA, $0424, $080D, $0809
!Tor_4Esc7_Colors_6a = $35B7, $2531, $14AB, $0424, $1000, $1000
!Tor_4Esc7_Colors_7a = $3198, $2112, $108B, $0004, $080D, $0809
!Tor_4Esc7_Colors_0b = $7FFF
!Tor_4Esc7_Colors_1b = $77BF
!Tor_4Esc7_Colors_2b = $739F
!Tor_4Esc7_Colors_3b = $6B5F
!Tor_4Esc7_Colors_4b = $673F
!Tor_4Esc7_Colors_5b = $5EFF
!Tor_4Esc7_Colors_6b = $5ADF
!Tor_4Esc7_Colors_7b = $529F

macro Tor_4Esc_List(n)
Tor_4Esc<n>_List:
  DW SetColorIndex, $00E8
Tor_4Esc<n>_List_Loop:
  DW $0002
    DW !Tor_4Esc0_Colors_0a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_0b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_1a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_1b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_2a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_2b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_3a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_3b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_4a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_4b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_5a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_5b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_6a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_6b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_7a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_7b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_6a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_6b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_5a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_5b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_4a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_4b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_3a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_3b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_2a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_2b
    DW GlowYeild
  DW $0002
    DW !Tor_4Esc0_Colors_1a
    DW SkipColors_4
    DW !Tor_4Esc0_Colors_1b
    DW GlowYeild
  DW GlowJMP, Tor_4Esc<n>_List_Loop
endmacro

%Tor_4Esc_List(0)
%Tor_4Esc_List(1)
%Tor_4Esc_List(2)
%Tor_4Esc_List(3)
%Tor_4Esc_List(4)
%Tor_4Esc_List(5)
%Tor_4Esc_List(6)
%Tor_4Esc_List(7)
print pc

org $8DF765
SkyFlash:
  DW $00BB, SkyFlashTable
org $8DF76D
WS_Green:
  DW $00BB, WS_GreenTable
org $8DF775
Blue_BG_:
  DW $00BB, Blue_BG_Table
SpoSpoBG:
  DW $00BB, SpoSpoBGTable
Purp_BG_:
  DW $00BB, Purp_BG_Table
Beacon__:
  DW $00BB, Beacon__Table
NorHot1_:
  DW $00BB, NorHot1_Table
NorHot2_:
  DW $00BB, NorHot2_Table
NorHot3_:
  DW $00BB, NorHot3_Table
NorHot4_:
  DW $00BB, NorHot4_Table
org $8DF79D
Waterfal:
  DW $00BB, WaterfalTable
Tourian_:
  DW $00BB, Tourian_Table
org $8DFFCD
Tor_2Esc:
  DW $00BB, Tor_2EscTable
Tor_3Esc:
  DW $00BB, Tor_3EscTable
Tor_4Esc:
  DW $00BB, Tor_4EscTable
OldT1Esc:
  DW $00BB, OldT1EscTable
OldT2Esc:
  DW $00BB, OldT2EscTable
OldT3Esc:
  DW $00BB, OldT3EscTable
SurfcEsc:
  DW $00BB, SurfcEscTable
Sky_Esc_:
  DW $00BB, Sky_Esc_Table
