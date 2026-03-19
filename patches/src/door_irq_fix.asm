;;; $83D7: Spawn hard-coded PLM ;;;
; certain maprando conditions can cause the hw door math to be interrupted by IRQ.
; this code wraps the doormath section with a disable irq.

lorom

org $8483d7
  phb
  phy
  phx
  phk
  plb
  ldy #$004e
loop:
  lda $1c37,y
  beq found
  dey
  dey
  bpl loop
  lda $06,s
  clc
  adc #$0004
  sta $06,s
  sec
  bra finished
found:
  sep #$20
  lda $08,s
  pha
  plb
  tyx
  ldy #$0002
  sei
  lda ($06,s),y
  sta $4202
  lda $07a5
  sta $4203
  ldy #$0001
  lda ($06,s),y
  rep #$20
  and #$00FF
  clc
  adc $4216
  asl a
  cli
  sta $1c87,x
  ldy #$0003
  lda ($06,s),y
  txy
  tax
  lda $06,s
  clc
  adc #$0004
  sta $06,s
  phk
  plb
  txa
  sta $1c37,y
  tyx
  tay
  lda #$0000
  sta $1dc7,x
  sta $7edf0c,x
  lda #$8469
  sta $1cd7,x
  lda $0002,y
  sta $1d27,x
  lda #$0001
  sta $7ede1c,x
  lda #$8da0
  sta $7ede6c,x
  stz $1d77,x
  stx $1c27
  tyx
  ldy $1c27
  jsr ($0000,x)
  clc
finished:
  plx
  ply
  plb
  rtl

assert pc() <= $848469