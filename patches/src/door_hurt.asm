lorom

; hijack door transition code that runs at end of fade-out:
org $82E2F7 
    jsr door_hurt

; Free space in bank $82:
org $82FA00
door_hurt:
    jsr $DE12  ; run hi-jacked instruction

    lda #$0005
    cmp $09D2 ; is x-ray selected?
    bne .skip ; if not, skip getting hurt through the door

    lda #$0005
    sta $18AA  ; set knockback timer to 5
    dec $09C2  ; decrease Samus' energy by 1
    
    ; extra stuff to make vertical setups work:
    stz $0B2E  ; set Samus Y speed to 0
    stz $0B2C  ; set Samus Y subspeed to 0
    lda #$0000
    sta $0A11  ; set previous movement type to standing
    sta $0A1F  ; set movement type to standing

.skip:
    rts
