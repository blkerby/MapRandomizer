arch snes.cpu
lorom

; In the Tourian Blue Hopper Room, move the left-most hopper a little to the right,
; to give Samus a chance to avoid a hit, which otherwise can be fatal in early game 
; completely outside the control of the player.
org $A1E387
    dw $D9FF, $00F8, $0061, $0000, $2000, $0000, $8000, $0000
    dw $D9FF, $00C6, $0099, $0000, $2000, $0000, $0000, $0000
    dw $FFFF
    db $02
