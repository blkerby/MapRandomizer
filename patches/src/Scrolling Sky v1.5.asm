lorom

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;
; Custom Layer 2 scrolling sky
;
; Warning: this asm overwrites the built in scrolling sky routines. You will need to manually add a layer 2 and data in it to rooms that use scrolling sky.
; Rooms that use the alternate scrolling sky (ex. East Ocean) will crash as that is removed entirely.
;
; Use 91C9 for the room's setup asm (aka Layer1_2)
; Use C116 for the room's main asm. (aka FX2)
;
; Put the scrolling sky BG on a custom layer 2 for the room in the first olumn of screens. The rest of the screens aren't used.
; Edit/Add a scroll table to match the positions of the clouds / scrolling sections.
; Note: If the room goes past the end of the table, the remaining sections will inherit their X scroll properties from the room. (Still fixed to one column.)
; Set the first tile's tile number to the scroll table index you want to use (000, 002, 004, etc.) this should be something in the CRE if it's sane.
; Note: The first row of tiles can never be visable.
; Note: you can save space in the rom if you copy the scrolling sky to every column of layer 2. It will compress nicely.
;

!YPositionReference = $7EFDFE

org $8F91C9 ;Normal scrolling sky room setup code
  JSL RoomSetupASM
  RTS

org $8FC116 ;Normal scrolling sky room main asm
  JSL RoomMainASM
  RTS

org $8FC120
  JSL RoomMainASM

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Add new tables here for each scrolling sky pattern needed
;
org $88F600 ;free space
ScrollingSkySectionTable_List:
  DW ScrollingSkySectionTable_000, ScrollingSkySectionTable_002, ScrollingSkySectionTable_004, ScrollingSkySectionTable_006
  DW ScrollingSkySectionTable_008, ScrollingSkySectionTable_00A, ScrollingSkySectionTable_00C, ScrollingSkySectionTable_00E
  DW ScrollingSkySectionTable_010

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; The first section should start at the top.
; Each next section must appear in order from top to bottom with increasing pixels locations.
; The last row should indicate the end of the last section and then have a null pointer for where to store the scroll value.
; The scroll values take 4 bytes to store. Only the highest word is actually read, so if you feel like doing something tricky you could read something else if you set the speed to be 0.
; If you want to make the table really long, you'll have to allocate more space in bank $7F for storing scroll values.
; Columns: top pixel row of the section, sub-pixels/frame to move (pixels start at the high word), section scroll storage address
;
!repeated = "$00000000"
ScrollingSkySectionTable_000: ;optimized vanilla (5 screens)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0010 : DD $0000C000 : DW $9F84
  DW $0038 : DD !repeated : DW $9F80
  DW $00D0 : DD !repeated : DW $9F84
  DW $00E0 : DD !repeated : DW $9F80
  DW $0120 : DD !repeated : DW $9F84
  DW $01A0 : DD !repeated : DW $9F80
  DW $01D8 : DD !repeated : DW $9F84
  DW $0238 : DD $00014000 : DW $9F88
  DW $0268 : DD !repeated : DW $9F84
  DW $02A0 : DD !repeated : DW $9F80
  DW $02E0 : DD !repeated : DW $9F88
  DW $0300 : DD !repeated : DW $9F80
  DW $0320 : DD !repeated : DW $9F84
  DW $0350 : DD !repeated : DW $9F80
  DW $0378 : DD !repeated : DW $9F84
  DW $03C8 : DD !repeated : DW $9F80
  DW $0440 : DD $00007000 : DW $9F8C
  DW $0460 : DD !repeated : DW $9F84
  DW $0480 : DD !repeated : DW $9F80
  DW $0490 : DD $00000000 : DW $9F90
  DW $0500 : DD $00000000 : DW $0000

ScrollingSkySectionTable_002: ;optimized vanilla (4 screens)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0020 : DD $0000C000 : DW $9F84
  DW $00A0 : DD !repeated : DW $9F80
  DW $00D8 : DD !repeated : DW $9F84
  DW $0138 : DD $00014000 : DW $9F88
  DW $0168 : DD $0000C000 : DW $9F84
  DW $01A0 : DD !repeated : DW $9F80
  DW $01E0 : DD !repeated : DW $9F88
  DW $0200 : DD !repeated : DW $9F80
  DW $0220 : DD !repeated : DW $9F84
  DW $0250 : DD !repeated : DW $9F80
  DW $0278 : DD !repeated : DW $9F84
  DW $02C8 : DD !repeated : DW $9F80
  DW $0340 : DD $00007000 : DW $9F8C
  DW $0360 : DD !repeated : DW $9F84
  DW $0380 : DD !repeated : DW $9F80
  DW $0390 : DD $00000000 : DW $9F90
  DW $0400 : DD $00000000 : DW $0000

ScrollingSkySectionTable_004: ;optimized vanilla (3 screens)
  DW $0000 : DD $0000C000 : DW $9F84
  DW $0038 : DD $00014000 : DW $9F88
  DW $0068 : DD !repeated : DW $9F84
  DW $00A0 : DD $00008000 : DW $9F80
  DW $00E0 : DD !repeated : DW $9F88
  DW $0100 : DD !repeated : DW $9F80
  DW $0120 : DD !repeated : DW $9F84
  DW $0150 : DD !repeated : DW $9F80
  DW $0178 : DD !repeated : DW $9F84
  DW $01C8 : DD !repeated : DW $9F80
  DW $0240 : DD $00007000 : DW $9F8C
  DW $0260 : DD !repeated : DW $9F84
  DW $0280 : DD !repeated : DW $9F80
  DW $0290 : DD $00000000 : DW $9F90
  DW $0300 : DD $00000000 : DW $0000

ScrollingSkySectionTable_006: ;optimized vanilla (2 screens)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0020 : DD $0000C000 : DW $9F84
  DW $0050 : DD !repeated : DW $9F80
  DW $0078 : DD !repeated : DW $9F84
  DW $00C8 : DD !repeated : DW $9F80
  DW $0140 : DD $00007000 : DW $9F8C
  DW $0160 : DD !repeated : DW $9F84
  DW $0180 : DD !repeated : DW $9F80
  DW $0190 : DD $00000000 : DW $9F90
  DW $0200 : DD $00000000 : DW $0000

ScrollingSkySectionTable_008: ;optimized vanilla (1 screen)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0040 : DD $00007000 : DW $9F8C
  DW $0060 : DD $0000C000 : DW $9F84
  DW $0080 : DD !repeated : DW $9F80
  DW $0090 : DD $00000000 : DW $9F90
  DW $0118 : DD $00000000 : DW $0000

ScrollingSkySectionTable_00A: ;optimized vanilla (6 screens)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0048 : DD $0000C000 : DW $9F84
  DW $0060 : DD !repeated : DW $9F80
  DW $00A8 : DD !repeated : DW $9F84
  DW $00E0 : DD !repeated : DW $9F80
  DW $0110 : DD !repeated : DW $9F84
  DW $0138 : DD !repeated : DW $9F80
  DW $01D0 : DD !repeated : DW $9F84
  DW $01E0 : DD !repeated : DW $9F80
  DW $0220 : DD !repeated : DW $9F84
  DW $02A0 : DD !repeated : DW $9F80
  DW $02D8 : DD !repeated : DW $9F84
  DW $0338 : DD $00014000 : DW $9F88
  DW $0368 : DD !repeated : DW $9F84
  DW $03A0 : DD !repeated : DW $9F80
  DW $03E0 : DD !repeated : DW $9F88
  DW $0400 : DD !repeated : DW $9F80
  DW $0420 : DD !repeated : DW $9F84
  DW $0450 : DD !repeated : DW $9F80
  DW $0478 : DD !repeated : DW $9F84
  DW $04C8 : DD !repeated : DW $9F80
  DW $0540 : DD $00007000 : DW $9F8C
  DW $0560 : DD !repeated : DW $9F84
  DW $0580 : DD !repeated : DW $9F80
  DW $0590 : DD $00000000 : DW $9F90
  DW $0600 : DD $00000000 : DW $0000

ScrollingSkySectionTable_00C: ;optimized vanilla (7 screens)
  DW $0000 : DD $00008000 : DW $9F80
  DW $0010 : DD $0000C000 : DW $9F84
  DW $0038 : DD !repeated : DW $9F80
  DW $00D0 : DD !repeated : DW $9F84
  DW $00E0 : DD !repeated : DW $9F80
  DW $0100 : DD !repeated : DW $9F80
  DW $0148 : DD !repeated : DW $9F84
  DW $0160 : DD !repeated : DW $9F80
  DW $01A8 : DD !repeated : DW $9F84
  DW $01E0 : DD !repeated : DW $9F80
  DW $0210 : DD !repeated : DW $9F84
  DW $0238 : DD !repeated : DW $9F80
  DW $02D0 : DD !repeated : DW $9F84
  DW $02E0 : DD !repeated : DW $9F80
  DW $0320 : DD !repeated : DW $9F84
  DW $03A0 : DD !repeated : DW $9F80
  DW $03D8 : DD !repeated : DW $9F84
  DW $0438 : DD $00014000 : DW $9F88
  DW $0468 : DD !repeated : DW $9F84
  DW $04A0 : DD !repeated : DW $9F80
  DW $04E0 : DD !repeated : DW $9F88
  DW $0500 : DD !repeated : DW $9F80
  DW $0520 : DD !repeated : DW $9F84
  DW $0550 : DD !repeated : DW $9F80
  DW $0578 : DD !repeated : DW $9F84
  DW $05C8 : DD !repeated : DW $9F80
  DW $0640 : DD $00007000 : DW $9F8C
  DW $0660 : DD !repeated : DW $9F84
  DW $0680 : DD !repeated : DW $9F80
  DW $0690 : DD $00000000 : DW $9F90
  DW $0678 : DD $00000000 : DW $0000

ScrollingSkySectionTable_00E: ;optimized vanilla (2 screens) -3 tiles
  DW $0000 : DD $00008000 : DW $9F80
  DW $0020 : DD $0000C000 : DW $9F84
  DW $0050 : DD !repeated : DW $9F80
  DW $0078 : DD !repeated : DW $9F84
  DW $00C8 : DD !repeated : DW $9F80
  DW $0110 : DD $00007000 : DW $9F8C
  DW $0130 : DD !repeated : DW $9F84
  DW $0150 : DD !repeated : DW $9F80
  DW $0160 : DD $00000000 : DW $9F90
  DW $01D0 : DD $00000000 : DW $0000

ScrollingSkySectionTable_010: ;optimized vanilla (4 screens) -3 tiles
  DW $0000 : DD $00008000 : DW $9F80
  DW $0020 : DD $0000C000 : DW $9F84
  DW $00A0 : DD !repeated : DW $9F80
  DW $00D8 : DD !repeated : DW $9F84
  DW $0138 : DD $00014000 : DW $9F88
  DW $0168 : DD $0000C000 : DW $9F84
  DW $01A0 : DD !repeated : DW $9F80
  DW $01E0 : DD !repeated : DW $9F88
  DW $0200 : DD !repeated : DW $9F80
  DW $0220 : DD !repeated : DW $9F84
  DW $0250 : DD !repeated : DW $9F80
  DW $0278 : DD !repeated : DW $9F84
  DW $02C8 : DD !repeated : DW $9F80
  DW $0310 : DD $00007000 : DW $9F8C
  DW $0330 : DD !repeated : DW $9F84
  DW $0350 : DD !repeated : DW $9F80
  DW $0360 : DD $00000000 : DW $9F90
  DW $03D0 : DD $00000000 : DW $0000

warnpc $88FFFF

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Room Setup ASM
; A JSR->JSL shim at $8F91C9 call this.
; Creates the HDMA object that handles changing the x scroll based on the position of the screen
;
org $88A7D8
RoomSetupASM:
  PHP
  SEP #$30
  LDA $091B
  ORA #$01
  STA $091B ;disable normal X scrolling graphics routines for layer 2
  LDA $091C
  ORA #$01
  STA $091C ;disable normal Y scrolling graphics routines for layer 2
  JSL $888435 ;spawn HDMA object
    DB $42, $0F
    DW ScrollInstructionList
  REP #$30
  LDA #$00E0
  STA $059A
  STZ $059C
  LDA $0915 ;Screen's Y position in pixels
  STA $0919 ;Screen's relative Y position reference point in pixels
  STA $B7 ;Value for $2110 (Y scroll of BG 2)
  STZ $0923 ;Screen's relative Y position offset for transitions
  JSL LoadFullBG
  LDA #$FFFF
  STA !YPositionReference
  PLP
  RTL
warnpc $88A81C

org $88AD76
ScrollInstructionList:
  DW $8655 ;HDMA table bank = $7E
    DB $7E
  DW $866A ;Indirect HDMA data bank = $7E
    DB $7E
  DW $8570
    DL ScrollPreInstruction ;$88ADB2 ; Pre-instruction
ScrollInstructionList_Loop:
  DW $7000, $9F00
  DW $85EC, ScrollInstructionList_Loop

ScrollPreInstruction:
  SEP #$30
  LDA $0A78
  BEQ ScrollPreInstruction_NotPaused
  RTL
ScrollPreInstruction_NotPaused:
  LDA #$4A
  STA $59 ;BG2 tilemap base address = $4800, size = 32x64
  REP #$30
  LDA $7F9602 ;First 16x16 tile of layer 2
  AND #$03FE ;Use to look up which table to use
  TAX
  LDA ScrollingSkySectionTable_List,x
  TAY
  LDA $0006,y ;HDMA table entry address
UpdateScrollPositions_Loop:
  TAX
  LDA $0002,y ;sub-pixel/frame
  CLC
  ADC $7E0000,x
  STA $7E0000,x
  LDA $0004,y ;pixel/frame
  ADC $7E0002,x
  STA $7E0002,x
  TYA
  CLC
  ADC #$0008
  TAY
  LDA $0006,y ;HDMA table entry address
  BNE UpdateScrollPositions_Loop ;look for terminator

  LDA #$0000
  STA $7E9FD8
  STA $7E9FDA
  LDA #$001F
  STA $7E9F00
  LDA #$059E
  STA $7E9F01 ;first entry in the HDMA table covers the section under the hud with a fixed scroll value

  LDA $0915 ;Screen's Y position in pixels
  CLC
  ADC #$0020
  STA $12 ;current first line (1 pixel below the hud)
  CLC
  ADC #$00C0
  STA $14 ;last line

  LDA $7F9602 ;First 16x16 tile of layer 2
  AND #$03FE ;Use to look up which table to use
  TAX
  LDA ScrollingSkySectionTable_List,x
  TAY ;Y = scrolling sky table index
  LDX #$0003 ;X = indirect HDMA table index starting at the second entry

BuildIndirectHDMATable_Loop:
  LDA $12 ;current first line
  CMP $0000,y ;top of scrolling section
  BMI BuildIndirectHDMATable_NotInView
  CMP $0008,y ;top of next scrolling section
  BMI ScrollingSection ;if the current first line in [top, top of next), go process this section
BuildIndirectHDMATable_NotInView:
  TYA
  CLC
  ADC #$0008
  TAY
  LDA $0006,y ;HDMA table entry address
  BNE BuildIndirectHDMATable_Loop ;look for terminator

  ;If we went past the bottom of the table we should treat the rest as a normal scrolling section
  ;Setup an entry to copy in the normal scroll value once and then leave the rest of the room below alone.
  LDA #$0001
  STA $7E9F00,x
  LDA #$00B5
  STA $7E9F01,x
  LDA #$0000
  STA $7E9F03,x ;terminate table
  RTL

ScrollingSection:
  LDA $0008,y ;top of next scrolling section
  SEC
  SBC $12 ;current first line
ScrollingSection_Loop:
  STA $18 ;section height
  CMP #$0080
  BMI ScrollingSection_Last
  LDA #$007F
  STA $7E9F00,x
  LDA $0006,y ;HDMA table entry address
  INC A
  INC A
  STA $7E9F01,x
  LDA $12
  CLC
  ADC #$007F
  STA $12
  INX
  INX
  INX
  LDA $18
  SEC
  SBC #$007F
  BRA ScrollingSection_Loop

ScrollingSection_Last:
  STA $7E9F00,x
  LDA $0006,y ;HDMA table entry address
  INC A
  INC A
  STA $7E9F01,x
  LDA $18 ;section height
  CLC
  ADC $12 ;current first line
  STA $12
  INX
  INX
  INX
  LDA $12 ;current first line
  CMP $14 ;last line
  BPL ScrollingSection_Exit
  JMP BuildIndirectHDMATable_Loop
ScrollingSection_Exit:
  LDA #$0000
  STA $7E9F03,x ;terminate table
  RTL

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Room Main ASM
; A JSR->JSL shim at $8FC116 calls this.
; Copies the graphics for two 8x8 rows above and two 8x8 rows below the screen.
; Data is loaded from the first column of screens in the layer 2 level data.
;
RoomMainASM:
  LDA $0A78 ;Pause timer, due to reserve tank and x-ray scope.
  BEQ RoomMainASM_NotPaused
  LDA #$0000
  STA $7E9F00
  RTL

DontUpdateTiles:
  LDA $0915 ;Screen's Y position in pixels
  STA $B7 ;Value for $2110 (Y scroll of BG 2)
  RTL

RoomMainASM_NotPaused:
  REP #$30
  LDA $0915 ;Screen's Y position in pixels
  AND #$FFF8
  CMP !YPositionReference
  BEQ DontUpdateTiles
  BIT #$0008
  BNE HalfScroll
  JSL CopyTilesFullScroll
  BRA SetupGraphicsDMA
HalfScroll:
  JSL CopyTilesHalfScroll

SetupGraphicsDMA:
  LDX $0330 ;'Stack' pointer for 00D0 table
  LDA #$0040
  ;Table of entries to update graphics, 7 bytes: Size (2 bytes), source address (3 bytes), VRAM target (2 bytes)
  STA $D0,x ;Graphics copy size 0
  STA $D7,x ;Graphics copy size 1

  LDA #$FB02
  STA $D2,x ;Graphics copy source address 0
  LDA #$FB42
  STA $D9,x ;Graphics copy source address 1

  SEP #$20
  LDA #$7F
  STA $D4,x ;Graphics copy source bank 0
  STA $DB,x ;Graphics copy source bank 1
  REP #$20

  LDA $59 ;Value for $2108 with 'A'   BG 2 Address and Size
  AND #$00FC ;Screen Base Address
  XBA
  STA $00

  LDA $0915 ;Screen's Y position in pixels
  AND #$FFF8
  CMP !YPositionReference
  BPL +

  LDA $0915 ;Screen's Y position in pixels
  SEC
  SBC #$0010
  BRA ++
+
  LDA $0915
  CLC
  ADC #$00F0
++
  AND #$01F8
  ASL A
  ASL A
  CLC
  ADC $00
  CMP #$5000
  BMI +
  SEC
  SBC #$0800
+
  CMP #$3FFF
  BPL +
  CLC
  ADC #$0800
+
  STA $D5,x ;Graphics copy target address 0
  CLC
  ADC #$0020
  CMP #$5000
  BMI +
  SEC
  SBC #$0800
+
  CMP #$3FFF
  BPL +
  CLC
  ADC #$0800
+
  STA $DC,x ;Graphics copy target address 1

SetupGraphicsDMA_Exit:
  TXA
  CLC
  ADC #$000E
  STA $0330 ;'Stack' pointer for 00D0 table

  LDA $0915 ;Screen's Y position in pixels
  STA $B7 ;Value for $2110 (Y scroll of BG 2)
  AND #$FFF8
  STA !YPositionReference
  RTL
print pc
warnpc $88B057

org $8AB180 ;Overwriting the tilemap that used to be used by scrolling sky
Copy8x8TileRowTop:
  LDA $7F9602,x
  PHX
  PHA
  AND #$03FF
  ASL A
  ASL A
  ASL A
  TAX

  PLA
  BIT #$0400
  BNE Copy8x8TileRowTop_HorizontalFlip
  BIT #$0800
  BNE Copy8x8TileRowTop_VerticalFlip

Copy8x8TileRowTop_NoFlip:
  LDA $7EA000,x
  PHX
  TYX
  STA $7FFB02,x
  PLX
  LDA $7EA002,x
  TYX
  STA $7FFB04,x

Copy8x8TileRowTop_Next:
  PLX
  INX
  INX
  INY
  INY
  INY
  INY
  CPY $00
  BMI Copy8x8TileRowTop
  RTS

Copy8x8TileRowTop_HorizontalFlip:
  BIT #$0800
  BNE Copy8x8TileRowTop_BothFlip
  LDA $7EA002,x
  PHX
  TYX
  EOR #$4000
  STA $7FFB02,x
  PLX
  LDA $7EA000,x
  TYX
  EOR #$4000
  STA $7FFB04,x
  BRA Copy8x8TileRowTop_Next

Copy8x8TileRowTop_VerticalFlip:
  LDA $7EA004,x
  PHX
  TYX
  EOR #$8000
  STA $7FFB02,x
  PLX
  LDA $7EA006,x
  TYX
  EOR #$8000
  STA $7FFB04,x
  BRA Copy8x8TileRowTop_Next

Copy8x8TileRowTop_BothFlip:
  LDA $7EA006,x
  PHX
  TYX
  EOR #$C000
  STA $7FFB02,x
  PLX
  LDA $7EA004,x
  TYX
  EOR #$C000
  STA $7FFB04,x
  BRA Copy8x8TileRowTop_Next


Copy8x8TileRowBottom:
  LDA $7F9602,x
  PHX
  PHA
  AND #$03FF
  ASL A
  ASL A
  ASL A
  TAX

  PLA
  BIT #$0400
  BNE Copy8x8TileRowBottom_HorizontalFlip
  BIT #$0800
  BNE Copy8x8TileRowBottom_VerticalFlip

Copy8x8TileRowBottom_NoFlip:
  LDA $7EA004,x
  PHX
  TYX
  STA $7FFB02,x
  PLX
  LDA $7EA006,x
  TYX
  STA $7FFB04,x

Copy8x8TileRowBottom_Next:
  PLX
  INX
  INX
  INY
  INY
  INY
  INY
  CPY $00
  BMI Copy8x8TileRowBottom
  RTS

Copy8x8TileRowBottom_HorizontalFlip:
  BIT #$0800
  BNE Copy8x8TileRowBottom_BothFlip
  LDA $7EA006,x
  PHX
  TYX
  EOR #$4000
  STA $7FFB02,x
  PLX
  LDA $7EA004,x
  TYX
  EOR #$4000
  STA $7FFB04,x
  BRA Copy8x8TileRowBottom_Next

Copy8x8TileRowBottom_VerticalFlip:
  LDA $7EA000,x
  PHX
  TYX
  EOR #$8000
  STA $7FFB02,x
  PLX
  LDA $7EA002,x
  TYX
  EOR #$8000
  STA $7FFB04,x
  BRA Copy8x8TileRowBottom_Next

Copy8x8TileRowBottom_BothFlip:
  LDA $7EA002,x
  PHX
  TYX
  EOR #$C000
  STA $7FFB02,x
  PLX
  LDA $7EA000,x
  TYX
  EOR #$C000
  STA $7FFB04,x
  BRA Copy8x8TileRowBottom_Next

CopyTilesFullScroll:
  LDA $0915 ;Screen's Y position in pixels
  AND #$FFF8
  CMP !YPositionReference
  BPL +

  LDA $0915 ;Screen's Y position in pixels
  LSR
  LSR
  LSR
  LSR
  DEC A
  LDY $07A5 ;Current room's width in tiles
  JSR FastMultiply
  ASL A
  TAX
  STA $14 ;position of first tile to copy 8x8 tiles from

  LDY #$0000
  LDA #$0040
  STA $00
  JSR Copy8x8TileRowTop
  LDX $14
  LDY #$0040
  LDA #$0080
  STA $00
  JSR Copy8x8TileRowBottom
  RTL

+
  LDA $0915 ;Screen's Y position in pixels
  LSR
  LSR
  LSR
  LSR
  CLC
  ADC #$000F ;one tile below screen
  LDY $07A5 ;Current room's width in tiles
  JSR FastMultiply
  ASL A
  TAX
  STA $14 ;position of first tile to copy 8x8 tiles from

  LDY #$0000
  LDA #$0040
  STA $00
  JSR Copy8x8TileRowTop
  LDX $14
  LDY #$0040
  LDA #$0080
  STA $00
  JSR Copy8x8TileRowBottom
  RTL


CopyTilesHalfScroll:
  LDA $07A5 ;Current room's width in tiles
  ASL
  STA $12

  LDA $0915 ;Screen's Y position in pixels
  AND #$FFF8
  CMP !YPositionReference
  BPL +

  LDA $0915
  LSR
  LSR
  LSR
  LSR
  DEC A
  LDY $07A5 ;Current room's width in tiles
  JSR FastMultiply
  ASL A
  TAX
  STA $14 ;position of first tile to copy 8x8 tiles from

  LDY #$0000
  LDA #$0040
  STA $00
  JSR Copy8x8TileRowBottom
  LDA $14
  CLC
  ADC $12
  TAX
  LDY #$0040
  LDA #$0080
  STA $00
  JSR Copy8x8TileRowTop
  RTL

+
  LDA $0915 ;Screen's Y position in pixels
  LSR
  LSR
  LSR
  LSR
  CLC
  ADC #$000F ;one tile below screen
  LDY $07A5 ;Current room's width in tiles
  JSR FastMultiply
  ASL A
  TAX
  STA $14 ;position of first tile to copy 8x8 tiles from

  LDY #$0000
  LDA #$0040
  STA $00
  JSR Copy8x8TileRowBottom
  LDA $14
  CLC
  ADC $12
  TAX
  LDY #$0040
  LDA #$0080
  STA $00
  JSR Copy8x8TileRowTop

  RTL

LoadFullBG:
  ;If this is a vertical transition, we need to load the bg where the screen will be, not where it is in the middle of the transition.
  ;If this is a horizontal transition, we want to use the current Y scroll value, which is after door has been aligned. Loading speed is halved.
  ;If loading from a save, we we want to use the current Y scroll value
  
  LDX $078D
  LDA $830006,X
  AND #$FF00
  STA $16
  LDA $0998 ;Game state
  CMP #$0006;Game is loading from save
  BEQ +
  CMP #$0028;Game is loading from demo
  BNE ++
+
  LDA $0915 ;Screen's Y position in pixels
  STA $16
++

  LDA #$FFF0
LoadFullBG_Loop:
  STA $02
  CMP #$0111
  BMI LoadFullBG_Continue
  JSR ExecuteDMA
  LDA #$FFFF
  STA !YPositionReference
  RTL

LoadFullBG_Continue:
  LDA $16
  CLC
  ADC $02
  LSR
  LSR
  LSR
  LSR
  LDY $07A5 ;Current room's width in tiles
  JSR FastMultiply
  ASL A
  TAX
  STA $14 ;position of first tile to copy 8x8 tiles from

  LDY #$0000
  LDA #$0040
  STA $00
  JSR Copy8x8TileRowTop
  LDX $14
  LDY #$0040
  LDA #$0080
  STA $00
  JSR Copy8x8TileRowBottom

  LDA $0791 ;Current room transition direction. 0 = right, 1 = left, 2 = down, 3 = up. +4 = Close a door on next screen
  BIT #$0002 ;is vertical transition?
  BEQ +
  LDA $14
  LSR
  CLC
  ADC $07A5 ;Current room's width in tiles
  ASL
  TAX
  STA $14

  LDY #$0080
  LDA #$00C0
  STA $00
  JSR Copy8x8TileRowTop
  LDX $14
  LDY #$00C0
  LDA #$0100
  STA $00
  JSR Copy8x8TileRowBottom
+

  LDX $0330 ;'Stack' pointer for 00D0 table
  LDA #$0040
  ;Table of entries to update graphics, 7 bytes: Size (2 bytes), source address (3 bytes), VRAM target (2 bytes)
  STA $D0,x ;Graphics copy size 0
  STA $D7,x ;Graphics copy size 1
  STA $DE,x ;Graphics copy size 2
  STA $E5,x ;Graphics copy size 3

  LDA #$FB02
  STA $D2,x ;Graphics copy source address 0
  LDA #$FB42
  STA $D9,x ;Graphics copy source address 1
  LDA #$FB82
  STA $E0,x ;Graphics copy source address 2
  LDA #$FBC2
  STA $E7,x ;Graphics copy source address 3

  SEP #$20
  LDA #$7F
  STA $D4,x ;Graphics copy source bank 0
  STA $DB,x ;Graphics copy source bank 1
  STA $E2,x ;Graphics copy source bank 2
  STA $E9,x ;Graphics copy source bank 3
  REP #$20

  LDA $59 ;Value for $2108 with 'A'   BG 2 Address and Size
  AND #$00FC ;Screen Base Address
  XBA
  STA $00

  LDA $16
  CLC
  ADC $02
  AND #$01F8
  ASL A
  ASL A
  CLC
  ADC $00
  STA $D5,x ;Graphics copy target address 0
  LDA $16
  CLC
  ADC $02
  CLC
  ADC #$0008
  AND #$01F8
  ASL A
  ASL A
  CLC
  ADC $00
  STA $DC,x ;Graphics copy target address 1
  LDA $16
  CLC
  ADC $02
  CLC
  ADC #$0010
  AND #$01F8
  ASL A
  ASL A
  CLC
  ADC $00
  STA $E3,x ;Graphics copy target address 2
  LDA $16
  CLC
  ADC $02
  CLC
  ADC #$0018
  AND #$01F8
  ASL A
  ASL A
  CLC
  ADC $00
  STA $EA,x ;Graphics copy target address 3

  LDA $0791 ;Current room transition direction. 0 = right, 1 = left, 2 = down, 3 = up. +4 = Close a door on next screen
  BIT #$0002 ;is vertical transition?
  BEQ +
  TXA
  CLC
  ADC #$001C
  STA $0330 ;'Stack' pointer for 00D0 table
  JSR ExecuteDMA
  LDA #$0020
  CLC
  ADC $02
  JMP LoadFullBG_Loop
+
  TXA
  CLC
  ADC #$000E ;ADC #$001C
  STA $0330 ;'Stack' pointer for 00D0 table
  JSR ExecuteDMA
  LDA #$0010
  CLC
  ADC $02
  JMP LoadFullBG_Loop


ExecuteDMA:
  LDA $84
  AND #$0080
  BEQ ExecuteDMA_NoNMI
ExecuteDMA_NMI:
  JSL $808338 ;Wait for NMI
  RTS
ExecuteDMA_NoNMI:
  JSL $808C83 ;Process DMA Stack
  RTS

FastMultiply:
  SEP #$30
  STA $211B
  STZ $211B
  TYA
  LSR
  STA $211C
  REP #$30
  LDA $2134
  ASL
  RTS

print pc
warnpc $8AE980