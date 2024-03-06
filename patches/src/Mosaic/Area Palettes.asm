lorom

; Hook "load state header" to modify palette address
org $82DF1D
  JSL GetPalettePointer
  BRA +
  NOP : NOP : NOP
  NOP : NOP : NOP
+
  ;LDA $0007,X
  ;STA $07C7
  ;LDA $0006,X
  ;STA $07C6

org $A7DC71
  JSL LoadPhantoonTargetColor
  ;LDA $CA61,X
  ;TAY

org $ADF24B
  CMP #$0008
  BNE +
  SEC
  RTL
+
  JSL MBLightsOn
  CLC
  RTL
warnpc $ADF40B

org $8AC000
EnablePalettesFlag:
  ; Two bytes go here
  ; DW $F0F0 ; vanilla = $1478

org $8AC002
GetArea:
  ; Enable area palettes is either the flag is set in ROM or one of the debug events is set
  LDA EnablePalettesFlag
  CMP #$F0F0
  BEQ UseMapArea
  LDA $7ED824 ; event bits $20-27
  AND #$00FF
  BNE UseMapArea

;UseTilesetArea:
  LDX $7E07BB ; tileset index
  LDA $8F0003,X
  AND #$00FF
  TAX
  LDA StandardArea,X
  AND #$00FF
  RTS
UseMapArea:
  LDA $1F5B  ; map area
  ASL
  CLC
  ADC $1F5B
  AND #$00FF
  RTS

GetPalettePointer:
  JSR GetArea
  TAX
  LDA AreaPalettes+1,X
  STA $07C7 ; palette bank
  LDA AreaPalettes+0,X
  STA $12 ; palette base offset

  LDX $7E07BB ; tileset index
  LDA $8F0003,X
  AND #$00FF

  STA $14
  ASL $14
  ASL $14 ; $14 = tileset index * 4
  XBA
  CLC
  ADC $14 ; tileset index * $104
  ADC $12 ; can't overflow the bank because we don't allow the area palettes to cross banks
  STA $07C6

  RTL

AreaPalettes:
  DL AreaPalettes_0, AreaPalettes_1, AreaPalettes_2, AreaPalettes_3, AreaPalettes_4, AreaPalettes_5, AreaPalettes_6, AreaPalettes_7

StandardArea:
  DB $00*3, $00*3 ;Crateria Surface
  DB $00*3, $00*3 ;Inner Crateria
  DB $03*3, $03*3 ;Wrecked Ship
  DB $01*3, $01*3 ;Brinstar
  DB $01*3 ;Tourian Statues Access/Blue brinstar
  DB $02*3, $02*3 ;Norfair
  DB $04*3, $04*3 ;Maridia
  DB $05*3, $05*3 ;Tourian
  DB $06*3, $06*3, $06*3, $60*3, $60*3, $60*3 ;Ceres
  DB $00*3, $00*3, $00*3, $00*3, $00*3 ;Utility Rooms
  ;Bosses
  DB $01*3 ;Kraid
  DB $04*3 ;Draygon
  DB $04*3 ;Draygon
  DB $01*3 ;SpoSpo
  DB $03*3 ;Phantoon

; Calculate the [A]th transitional color from start color in [X] to target color in [Y]
; Copy of $82DAA6 but the current denominator is stored at $00
ComputeTransitionalColor:
  PHA
  PHA
  PHX
  PHY
  LDA $01,S
  AND #$001F
  TAY
  LDA $03,S
  AND #$001F
  TAX
  LDA $05,S
  JSR ComputeTransitionalComponent
  STA $07,S
  LDA $01,S
  ASL
  ASL
  ASL
  XBA
  AND #$001F
  TAY
  LDA $03,S
  ASL
  ASL
  ASL
  XBA
  AND #$001F
  TAX
  LDA $05,S
  JSR ComputeTransitionalComponent
  ASL
  ASL
  ASL
  ASL
  ASL
  ORA $07,S
  STA $07,S
  LDA $01,S
  LSR
  LSR
  XBA
  AND #$001F
  TAY
  LDA $03,S
  LSR
  LSR
  XBA
  AND #$001F
  TAX
  LDA $05,S
  JSR ComputeTransitionalComponent
  ASL
  ASL
  XBA
  ORA $07,S
  STA $07,S
  PLY
  PLX
  PLA
  PLA
  RTS
ComputeTransitionalComponent:
  CMP #$0000
  BNE +
  TXA
  RTS
+
  DEC
  CMP $00
  BNE +
  TYA
  RTS
+
  PHX
  INC
  STA $14
  TYA
  SEC
  SBC $01,S
  STA $12
  BPL +
  EOR #$FFFF
  INC
+
  SEP #$21
  STZ $4204
  STA $4205
  LDA $00
  SBC $14
  INC
  STA $4206
  REP #$20
  NOP
  NOP
  NOP
  NOP
  NOP
  LDA $4214
  BIT $12
  BPL +
  EOR #$FFFF
  INC
+
  STA $12
  PLA
  XBA
  CLC
  ADC $12
  XBA
  AND #$00FF
  RTS

LoadPhantoonTargetColor:
  TXY
  JSR GetArea
  TAX
  LDA AreaPalettes+1,X
  STA $13 ; palette bank
  LDA AreaPalettes+0,X
  CLC
  ADC #$0412 ; WS awake is palette $04 + skip header
  STA $12
  LDA [$12],Y
  TYX
  TAY
  RTL

MBLightsOn:
  INC
  STA $16 ; transition index
  LDA #$0007
  STA $00

  JSR GetArea
  TAX
  LDA AreaPalettes+1,X
  STA $03 ; palette bank
  LDA AreaPalettes+0,X
  CLC
  ADC #$0E3A ; MB is palette $0E + skip header
  STA $02

  LDA #$0062
  STA $06
-
  LDY $06
  LDA [$02],Y
  TAY ; target color
  LDX #$0000 ; source color
  LDA $16
  JSR ComputeTransitionalColor
  LDX $06
  STA $7EC000,X
  INX
  INX
  STX $06
  CPX #$0080
  BMI -

  LDA #$00A2
  STA $06
-
  LDY $06
  LDA [$02],Y
  TAY
  LDX #$0000
  LDA $16
  JSR ComputeTransitionalColor
  LDX $06
  STA $7EC000,X
  INX
  INX
  STX $06
  CPX #$00C0
  BMI -
  RTL

; Use "InputFile" working directory mode in SMART if you want this to assemble in xkas
; Each uncompressed paletter is 256 bytes. Compressing these palettes doesn't always make them smaller anyway

macro PaletteFile(t, n, area)
AreaPalettes_<n>_<t>:
  DB $E0, $FF ; header for the decompressor (copy $100 literal bytes)
incbin ..\..\<area>\Export\Tileset\SCE\<t>\palette.snes ; not actually compressed
  DB $FF ; footer for the decompressor
  DB $FF ; 1 byte padding
endmacro

macro PaletteSet(n, area)
!dir = ..\..\<area>\Export\Tileset\SCE
!file = palette.snes

print "Area Palettes <n>:"
print pc
AreaPalettes_<n>:
%PaletteFile(00, <n>, <area>)
%PaletteFile(01, <n>, <area>)
%PaletteFile(02, <n>, <area>)
%PaletteFile(03, <n>, <area>)
%PaletteFile(04, <n>, <area>)
%PaletteFile(05, <n>, <area>)
%PaletteFile(06, <n>, <area>)
%PaletteFile(07, <n>, <area>)
%PaletteFile(08, <n>, <area>)
%PaletteFile(09, <n>, <area>)
%PaletteFile(0A, <n>, <area>)
%PaletteFile(0B, <n>, <area>)
%PaletteFile(0C, <n>, <area>)
%PaletteFile(0D, <n>, <area>)
%PaletteFile(0E, <n>, <area>)
%PaletteFile(0F, <n>, <area>)
%PaletteFile(10, <n>, <area>)
%PaletteFile(11, <n>, <area>)
%PaletteFile(12, <n>, <area>)
%PaletteFile(13, <n>, <area>)
%PaletteFile(14, <n>, <area>)
%PaletteFile(15, <n>, <area>)
%PaletteFile(16, <n>, <area>)
%PaletteFile(17, <n>, <area>)
%PaletteFile(18, <n>, <area>)
%PaletteFile(19, <n>, <area>)
%PaletteFile(1A, <n>, <area>)
%PaletteFile(1B, <n>, <area>)
%PaletteFile(1C, <n>, <area>)
%PaletteFile(1D, <n>, <area>)
%PaletteFile(1E, <n>, <area>)
endmacro

org $C08000
%PaletteSet(0, CrateriaPalette)
%PaletteSet(1, BrinstarPalette)
%PaletteSet(2, NorfairPalette)
%PaletteSet(3, WreckedShipPalette)
warnpc $C0FFFF
org $C18000
%PaletteSet(4, MaridiaPalette)
%PaletteSet(5, TourianPalette)
%PaletteSet(6, CrateriaPalette)
%PaletteSet(7, CrateriaPalette)
warnpc $C1FFFF
