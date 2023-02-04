lorom

; Replace code to decrement ammo during rainbow beam
org $A9C4C4
    LDA #$0000
    STA $09D2   ; Clear HUD item
    RTS

;org $A9C544
;dw #$0000
