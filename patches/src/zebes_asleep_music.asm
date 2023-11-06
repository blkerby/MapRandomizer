lorom

!bank_82_freespace_start = $82FE80
!bank_82_freespace_end = $82FF00

; Hook room header loading routine, after songset/track have been set
org $82DF4A
    jsr music_load_hook

; Replace music tracks based on relevant events: Zebes awake, Phantoon dead, or have PBs
org !bank_82_freespace_start
music_load_hook:
    php

    ; Load songset/index into A
    sep #$20
    lda $07CB
    xba
    lda $07C9
    rep #$20

crateria_pirates_check:
    cmp #$0905  ; Is this Crateria Pirates track (Crateria subarea not containing Ship)
    bne return_to_crateria_check
    lda $7ED820
    bit #$0001  ; Is Zebes awake
    bne done

    lda #$0006
    sta $07CB
    lda $09D0  ; Max PB count == 0 ?
    beq done   ; Keep song index 5: No PBs -> Zebes asleep music with storm

    lda #$0007  ; PBs -> Zebes asleep music without storm
    sta $07C9
    jmp done

.zebes_awake:
;    ; Landing Site:
;    cmp #$0606
;    bne +
;    lda #$0005
;    sta $07C9
;    jmp .skip
;+

return_to_crateria_check:
    cmp #$0C05  ; Is this Return to Crateria track (Crateria subarea containing Ship)
    bne wrecked_ship_check  ; Skip if not Return to Crateria track
    lda $09D0  ; Max PB count != 0 ?
    bne done   ; PBs in Crateria subarea containing the Ship -> keep Return to Crateria track

    lda $7ED820
    bit #$0001  ; Is Zebes awake
    bne .zebes_awake

    ; Zebes asleep in Crateria subarea containing the Ship -> intro song
    lda #$0036
    sta $07CB
    lda #$0005  
    sta $07C9
    jmp done

.zebes_awake:
    lda #$0006 ; Zebes awake in Crateria subarea containing the Ship, without PBs -> plain storm track
    sta $07CB
    lda #$0006  
    sta $07C9
    jmp done

wrecked_ship_check:
    cmp #$3005  ; Is this Wrecked Ship track?
    bne done
    lda $7ed82b
    and #$0001
    beq done  ; Skip if Phantoon isn't dead

    ; Wrecked Ship track with Phantoon dead -> power-on Wrecked Ship track
    lda #$0006
    sta $07C9
;    jmp done

done:
    plp
    lda $0006,x  ; run hi-jacked instruction
    rts
warnpc !bank_82_freespace_end
