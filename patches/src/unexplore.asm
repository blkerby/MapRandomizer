arch snes.cpu
lorom

!ram_ctrl = $8B

; hijack code that sets tile to explored:
; hijacked instruction: LDA $07F7,x
org $90A97C
    jmp maybe_set_explored

; code in bank $90 free space:
org $90F63A
maybe_set_explored:
    lda !ram_ctrl
    and $09B8 : beq no_input  ; Check for item cancel
    lda !ram_ctrl
    and $09BE : beq no_input  ; Check for aim up

    sep #$20
    ; A <- 0xFF - (0x80 >> Y)
    lda #$FF
    clc
    sbc $AC04,y

    and $07F7, x
    jmp $A984  ; jump back to vanilla code

no_input:
    ; run hijacked instructions
    sep #$20
    lda $07F7,x

    ; jump back to vanilla code
    jmp $A981


