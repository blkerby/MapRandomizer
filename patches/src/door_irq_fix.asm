;;; $83D7: Spawn hard-coded PLM ;;;
; certain maprando conditions can cause the hw door math to be interrupted by IRQ.
; this code wraps the doormath section with a disable irq.

lorom

org $8483fc ; ldy #$0002
jsr disable_irq

org $848419 ; $84:8419 ; sta $1c87,x
jsr enable_irq


org $84F4A0
disable_irq:
  sei
  ldy #$0002
  rts

enable_irq:
  cli
  sta $1c87,x
  rts

assert pc() <= $84F500 ; 84F4AA