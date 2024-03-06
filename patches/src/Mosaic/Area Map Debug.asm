; This is here so we can test this stuff in Mosaic without having the map code in too
; We shouldn't need it in the main repo

; Set the "map area" based on the first event bit set in $20-27
; Set bit 20 for Crateria, etc
; If no bits are set, then always set the map area to match the tileset

lorom
org $82DEF7
  ;LDX $07BB
  JSR SetMapAreaInject
org $82BEA3 ; overwrite unused code
SetMapAreaInject:
  LDA $7ED824 ; event bits $20-27
  AND #$00FF
  BEQ MatchMapToTileset
  STZ $1F5B
  DEC $1F5B
-
  INC $1F5B
  LSR
  BCC -
  LDX $07BB
  RTS

MatchMapToTileset:
  LDX $07BB
  LDA $8F0003,X
  AND #$00FF
  TAX
  LDA.l StandardArea,X
  AND #$00FF
  STA $1F5B
  LDX $07BB
  RTS

StandardArea:
  DB $00, $00 ;Crateria Surface
  DB $00, $00 ;Inner Crateria
  DB $03, $03 ;Wrecked Ship
  DB $01, $01 ;Brinstar
  DB $01 ;Tourian Statues Access
  DB $02, $02 ;Norfair
  DB $04, $04 ;Maridia
  DB $05, $05 ;Tourian
  DB $06, $06, $06, $06, $06, $06 ;Ceres
  DB $00, $00, $00, $00, $00 ;Utility Rooms
  ;Bosses
  DB $01 ;Kraid
  DB $02 ;Crocomire
  DB $04 ;Draygon
  DB $01 ;SpoSpo
  DB $03 ;Phantoon

warnpc $82BF03