lorom

; Edit plms for uopening and closing teh stairs in bowling alley to not actually alter graphics

;org $8482BD
;  AND #$0F

org $84D3CF
  DW #OpenBowlingStairs
  DW $D3D7 ; Edit some other slopes without touching gfx
  DW $86BC ; delete

org $84D3EC
  DW #CloseBowlingStairs
  DW $D3F4 ; Edit some other slopes without touching gfx
  DW $86BC ; delete

org $849CC5 ;overwrite the old plm draw instructions
OpenBowlingStairs:
  JSL OpenBowlingStairs_2
  RTS
CloseBowlingStairs:
  JSL CloseBowlingStairs_2
  RTS
warnpc $849D59

org $8AB600 ; free space (due to scrolling sky)
OpenBowlingStairs_2:
  PHX
  PHY

  LDA #$0000
  LDX #$15EE
  LDY #$000D
-
  JSR UpdateTileType
  INX
  INX
  DEY
  BNE -
  LDA #$1000
  JSR UpdateTileType

  LDA #$0000
  LDX #$19AE
  LDY #$0008
-
  JSR UpdateTileType
  INX
  INX
  DEY
  BNE -
  LDA #$1000
  JSR UpdateTileType

  LDA #$0000
  LDX #$1A78
  JSR UpdateTileType
  LDA #$0000
  LDX #$1A7A
  JSR UpdateTileType

  LDA #$0000
  LDX #$1B38
  JSR UpdateTileType

  LDA #$1000
  LDX #$1BF8
  JSR UpdateTileType

  PLY
  PLX
  RTL
CloseBowlingStairs_2:
  PHX
  PHY

  LDA #$A000
  LDX #$15EE
  LDY #$000D
-
  JSR UpdateTileType
  INX
  INX
  DEY
  BNE -
  LDA #$8000
  JSR UpdateTileType

  LDA #$8000
  LDX #$19AE
  LDY #$0009
-
  JSR UpdateTileType
  INX
  INX
  DEY
  BNE -

  LDA #$8000
  LDX #$1A78
  JSR UpdateTileType
  LDA #$8000
  LDX #$1A7A
  JSR UpdateTileType

  LDA #$8000
  LDX #$1B38
  JSR UpdateTileType

  LDA #$8000
  LDX #$1BF8
  JSR UpdateTileType

  PLY
  PLX
  RTL

UpdateTileType:
  STA $12
  LDA $7F0002,X
  AND #$0FFF
  ORA $12
  STA $7F0002,X
  RTS
