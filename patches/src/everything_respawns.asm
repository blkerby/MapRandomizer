lorom

org $A0A3E0
  LDA #$4000

org $86EFA5
  LDA #$4000

org $A095F1 ; repair unused vanilla routine
  PHB
  PHX
  PHY
  PEA $A000
  PLB
  PLB
  REP #$30
  LDA $0E54
  LSR
  LSR
  CLC
  ADC $07CF
  TAX
  LDY $0E54
  LDA $A10000,X
  STA $0F78,Y
  LDA $A10002,X
  STA $0F7A,Y
  LDA $A10004,X
  STA $0F7E,Y
  LDA $A10006,X
  STA $0F92,Y
  LDA $A10008,X
  STA $0F86,Y
  LDA $A1000A,X
  STA $0F88,Y
  LDA $A1000C,X
  STA $0FB4,Y
  LDA $A1000E,X
  STA $0FB6,Y
  PHX
  TYX
  LDA $7E7008,X
  AND #$0E00
  STA $0F96,X
  LDA $7E7006,X
  AND #$01FF
  STA $0F98,X
  PLX
  LDA #$0000
  STA $0F9E,Y
  STA $0F9C,Y
  STA $0FA0,Y
  STA $0F90,Y
  STA $0FA4,Y
  STA $0FA8,Y
  STA $0FAA,Y
  STA $0FAC,Y
  STA $0FAE,Y
  STA $0FB0,Y
  STA $0FB2,Y
  LDA #$0001
  STA $0F94,Y
  LDX $0E54
  LDA $0F78,X
  TAX
  LDA $0012,X
  STA $1784
  LDA $0008,X
  STA $0F82,Y
  LDA $000A,X
  STA $0F84,Y
  LDA $0004,X
  STA $0F8C,Y
  LDA $0039,X
  AND #$00FF
  STA $0F9A,Y
  LDA $000C,X
  STA $0FA6,Y
  STA $1786
  JSL $A096BA
  PLY
  PLX
  PLB
  RTL

; boss dead state conditions
org $8FE606
  BRA $07
org $8FE633
  BRA $07

; Kraid
org $A7A934
  LDA #$0000

; Phantoon

; Draygon

; Ridley
org $A6A0FC
  LDA #$0000
org $A6C90E
  JML $A0922B

; Torizos
org $AAC886
  BRA $0D
org $84D60D
  BRA $06

; Spore Spawn
org $A5EA96
  LDA #$0000

; Crocomire
org $A48A7A 
  LDA #$0000

; Botwoon
org $B39587
  LDA #$0000

; Metroids
org $8FDAE1+13
  DB #16
org $8FDB31+13
  DB #16
org $8FDB7D+13
  DB #16
org $8FDBCD+13
  DB #16

; mocktroid
org $A3A789
  JSR $A942

; skree
org $A3C7CB
  JSL $A095F1
  RTL

; metal skree
org $A38AA8
  JSL $A095F1
  RTL

; boulder
org $A688B8
  LDA $0F7A,X
  STA $12
  LDA $0F7E,X
  STA $14
  LDA #$0011
  LDY #$E509
  JSL $868097
  LDA #$0043
  JSL $8090CB
  JSL $A095F1
  RTS
org $A68989
  LDA #$89FC
  STA $0FA8,X
  LDA #$0042
  JSL $8090CB
  LDA $0F7A,X
  STA $12
  LDA $0F7E,X
  STA $14
  LDA #$0011
  LDY #$E509
  JSL $868097
  LDA #$0043
  JSL $8090CB
  JSL $A095F1
  RTS

; boyon
org $A28751
  JSR ResetBoyon
org $A2F4B0
ResetBoyon:
  STZ $0FB2,X
  JSL ResetExtra
  RTS

ResetExtra:
  PHX
  PHY
  LDX $0E54
  LDY #$001F
  TDC
.loop
  STA $7E7800,X
  STA $7E8000,X
  INX
  INX
  DEY
  BPL .loop
  PLY
  PLX
  RTL

; zebitite
org $A6FC8D
  JML $A0922B
org $A6FCA2
  JSL $A0922B
