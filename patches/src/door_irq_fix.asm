;;; $83D7: Spawn hard-coded PLM ;;;
; certain maprando conditions can cause the hw door math to be interrupted by IRQ.
; this code wraps the doormath section with a disable irq.

lorom

org $8483fc ; ldy #$0002
jmp disable_irq

org $848419 ; $84:8419 ; sta $1c87,x
jmp enable_irq


org $84F4A0
disable_irq:
  sei
  ldy #$0002
  jmp $83FF

enable_irq:
  cli
  sta $1c87,x
  jmp $841C

assert pc() <= $84F500 ; 84F4AA