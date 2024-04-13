!bank_80_free_space_start = $80E180
!bank_80_free_space_end = $80E200

; vanilla version: (for if top and bottom weren't swapped like they are in SpriteSomething)
;org $8093A9
;    jsr hook_samus_dma_top_part_1
;
;org $8093C5
;    jsr hook_samus_dma_top_part_2
;
;org $8093F2
;    jsr hook_samus_dma_bottom_part_1
;
;org $80940E
;    jsr hook_samus_dma_bottom_part_2

org $8093A9
    jsr hook_samus_dma_bottom_part_1

org $8093C5
    jsr hook_samus_dma_bottom_part_2

org $8093F2
    jsr hook_samus_dma_top_part_1

org $80940E
    jsr hook_samus_dma_top_part_2


org !bank_80_free_space_start

hook_samus_dma_top_part_1:
    bne .non_zero
    lda #$0001      ; transfer a minimum of 1 byte, since 0 would be interpreted as $10000
    bra .end
.non_zero:
    cmp #$0400      ; transfer a maximum of $400 bytes, to stay inside the VRAM space reserved for Samus.
    bcc .end
    lda #$0400
.end:
    sta $4315   ; run hi-jacked instruction
    rts

hook_samus_dma_top_part_2:
    cmp #$0200      ; transfer a maximum of $200 bytes, to stay inside the VRAM space reserved for Samus.
    bcc .end
    lda #$0200
.end:
    sta $4315   ; run hi-jacked instruction
    rts

hook_samus_dma_bottom_part_1:
    bne .non_zero
    lda #$0001      ; transfer a minimum of 1 byte, since 0 would be interpreted as $10000
    bra .end
.non_zero:
    cmp #$0300      ; transfer a maximum of $300 bytes, to stay inside the VRAM space reserved for Samus.
    bcc .end
    lda #$0300
.end:
    sta $4315   ; run hi-jacked instruction
    rts

hook_samus_dma_bottom_part_2:
    cmp #$0100      ; transfer a maximum of $100 bytes, to stay inside the VRAM space reserved for Samus.
    bcc .end
    lda #$0100
.end:
    sta $4315   ; run hi-jacked instruction
    rts

warnpc !bank_80_free_space_end

