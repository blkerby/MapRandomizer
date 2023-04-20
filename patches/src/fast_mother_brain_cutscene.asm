lorom

;;; Eliminate delay before Mother Brain rising:

org $A98D73
    LDA #$0001          ; replaces: LDA #$0080 

org $A98D85
    LDA #$0001          ; replaces: LDA #$0020 

org $A98DAE 
    LDA #$0001          ; replaces: LDA #$0100  

;;; Shorten cutscene when Big Boy drains Mother Brain:

org $A9BECD
    CMP #$0004   ; was CMP #$0006

org $A9BF1E 
    CMP #$0006   ; was CMP #$0008

org $A9C049 
    dw $0008, $0008, $0008, $0008, $0008 ;, $0010, 0010, 0010

;; Shorten explosions at end of MB3

org $A9AF07
    dw $AF21

org $A9AF4E
    dw $0004

org $A9AFC3
    dw $0004

org $A9B00D
    dw $0004

;; Speed up the corpse fading & rotting

org $A9B19E
    LDA #$0008

org $A99D2D
    dw $0010

org $A99D31
    dw $0010

org $A9B1B2
    dw $0050

;; Hook to make Samus stand up before escape (in "Short" Mother Brain fight mode)

org $A9FD00
    LDA #$0017             ;\ Make Samus stand up
    JSL $90F084            ;/
    LDA #$0000             ;\
    JSL $808FC1            ;/ Queue music stop
;    LDA #$B173             ; Run hi-jacked instruction
    LDA #$B1D5
    RTS
warnpc $A9FD40