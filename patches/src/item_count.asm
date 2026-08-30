; Sum item bits to calculate collection percent.
; 
; credits % is displayed as a fraction if every item isn't placed.
; nn_357 / StagShot

!bank_8b_free_space_start = $8bf900
!bank_8b_free_space_end  = $8bfae9
!nothing_item_total = $dfff0e    ; overwritten by patch.rs, contains the sum of 'nothing'
!initial_item_bits = $b5fe12     ; modified by patch.rs, also used in new_game.asm
!item_count = $12

org !nothing_item_total
  dw $0000

;;; $E627: Instruction - draw item percentage count ;;;

org $8be627
itempercent:
  php
  phb
  phk
  plb
  rep #$30
  phx
  phy
  jsr countitems
  lda !nothing_item_total
  bne .fractional

  lda !item_count
  and #$00ff
  cmp #$0064
  bne .not_100
  
  lda $e745       ; 100% completion
  sta $7e339c
  lda $e747
  sta $7e33dc
  lda $e741
  sta $7e339e
  lda $e743
  sta $7e33de
  lda $e741
  sta $7e33a0
  lda $e743
  sta $7e33e0
  bra .percent_symbol

.not_100
  ldx #$0002
  jsr .digit2
  bra .percent_symbol

.fractional
  jsr load_fractional_tiles
  lda #$0064
  sec
  sbc !nothing_item_total
  sta $14

  lda !item_count
  and #$00ff
  ldx #$0000
  jsr .digit2

  lda $14
  ldx #$0006
  jsr .digit2

  lda #$3888        ; add custom '/' tile to tilemap
  sta $7e33a0
  lda #$3898
  sta $7e33e0
  bra .finished

.percent_symbol
  lda #$386a
  sta $7e33a2
  lda #$387a
  sta $7e33e2
.finished
  ply
  plx
  plb
  plp
  rts

.digit2
  sta $4204
  sep #$20
  lda #$0A
  sta $4206
  nop #6
  rep #$20
  lda $4214
  asl a
  asl a
  tay
  lda $e741,y
  sta $7e339c,x
  lda $e743,y
  sta $7e33dc,x
  lda $4216
  asl a
  asl a
  tay
  lda $e741,y
  sta $7e339e,x
  lda $e743,y
  sta $7e33de,x
  rts

assert pc() <= $8be741        ; Tilemap values for decimal digits:

org !bank_8b_free_space_start

countitems:
  php
  sep #$30

  lda #$00
  sta !item_count
  ldx #$00
  
.loop
  lda !initial_item_bits,x
  eor #$ff
  and $7ed870,x
  tay
  lda bitcounttable,y
  clc
  adc !item_count
  sta !item_count

  inx
  cpx #$14            
  bne .loop

  plp
  rts

load_fractional_tiles:
  php
  sep #$30
WaitVB:
  lda $4212
  and #$80
  beq WaitVB

  lda #$80        ; top_slash → $4880
  sta $2115
  lda #$80
  sta $2116
  lda #$48
  sta $2117

  lda #$01
  sta $4300
  lda #$18
  sta $4301
  lda.b #top_slash
  sta $4302
  lda.b #top_slash>>8
  sta $4303
  lda #$8b
  sta $4304
  lda #$20
  sta $4305
  stz $4306
  lda #$01
  sta $420B

  lda #$80        ; bottom_slash → $4980
  sta $2116
  lda #$49
  sta $2117

  lda.b #bottom_slash
  sta $4302
  lda.b #bottom_slash>>8
  sta $4303
  lda #$20
  sta $4305
  stz $4306
  lda #$01
  sta $420B
  
  plp
  rts

top_slash:
  db $00, $07, $02, $0D, $06, $09, $06, $09 
  db $0C, $13, $0C, $32, $18, $26, $18, $24
  db $00, $00, $00, $00, $00, $00, $00, $00
  db $00, $00, $00, $00, $00, $00, $00, $00

bottom_slash:
  db $18, $64, $30, $4C, $30, $C8, $60, $98 
  db $60, $90, $40, $B0, $00, $E0, $00, $00
  db $00, $00, $00, $00, $00, $00, $00, $00
  db $00, $00, $00, $00, $00, $00, $00, $00

bitcounttable:
  db $00,$01,$01,$02,$01,$02,$02,$03,$01,$02,$02,$03,$02,$03,$03,$04
  db $01,$02,$02,$03,$02,$03,$03,$04,$02,$03,$03,$04,$03,$04,$04,$05
  db $01,$02,$02,$03,$02,$03,$03,$04,$02,$03,$03,$04,$03,$04,$04,$05
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $01,$02,$02,$03,$02,$03,$03,$04,$02,$03,$03,$04,$03,$04,$04,$05
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $03,$04,$04,$05,$04,$05,$05,$06,$04,$05,$05,$06,$05,$06,$06,$07
  db $01,$02,$02,$03,$02,$03,$03,$04,$02,$03,$03,$04,$03,$04,$04,$05
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $03,$04,$04,$05,$04,$05,$05,$06,$04,$05,$05,$06,$05,$06,$06,$07
  db $02,$03,$03,$04,$03,$04,$04,$05,$03,$04,$04,$05,$04,$05,$05,$06
  db $03,$04,$04,$05,$04,$05,$05,$06,$04,$05,$05,$06,$05,$06,$06,$07
  db $03,$04,$04,$05,$04,$05,$05,$06,$04,$05,$05,$06,$05,$06,$06,$07
  db $04,$05,$05,$06,$05,$06,$06,$07,$05,$06,$06,$07,$06,$07,$07,$08

assert pc() <= !bank_8b_free_space_end