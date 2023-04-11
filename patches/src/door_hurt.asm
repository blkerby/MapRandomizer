lorom

; hijack door transition code that runs at end of fade-out:
org $82E2F7 
    jsr door_hurt

; Free space in bank $82:
org $82FA00
door_hurt:
    lda #$0005
    sta $18AA  ; set knockback timer to 5
    dec $09C2  ; decrease Samus' energy by 1
    jsr $DE12  ; run hi-jacked instruction
    rts
