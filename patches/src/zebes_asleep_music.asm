lorom

!bank_82_freespace_start = $82FE80
!bank_82_freespace_end = $82FF00

; Hook room header loading routine, after songset/track have been set
org $82DF4A
    jsr music_load_hook

; Replace Crateria tracks with Zebes asleep versions where applicable
org !bank_82_freespace_start
music_load_hook:
    php
    lda $7ED820
    bit #$0001
    bne .skip

    sep #$20
    lda $07CB
    xba
    lda $07C9
    rep #$20

    ; Landing Site:
    cmp #$0606
    bne +
    lda #$0005
    sta $07C9
    jmp .skip
+

    ; Crateria Pirates:
    cmp #$0905
    bne +
    lda #$0006
    sta $07CB
    jmp .skip
+

    ; Return to Crateria (Crateria subarea not containing the Ship)
    cmp #$0C05
    bne +
    lda #$0006
    sta $07CB
    lda #$0007
    sta $07C9
+

.skip:
    plp
    lda $0006,x  ; run hi-jacked instruction
    rts
warnpc !bank_82_freespace_end
