lorom

;;;
; Shorten Big Boy cutscene:
;;;

; Delay before Big Boy attacks Dead Sidehopper
org $A9F031
    LDA #$002C          ; replaces: LDA #$01D0

; Hop initial velocities (make the first hop bigger, to effective skip a hop)
org $A9D951
    dw $FE00, $FE00, $FC00, $FE00
    dw $0120, $0250, $0300, $01C0

; Delay between Dead Sidehopper hops
org $A9D916
    LDA #$0001          ; replaces: LDA #$0040

; X position that Big Boy rushes toward
org $A9F049
    LDA #$01B0          ; replaces: LDA #$0248

; Big Boy realizing what he did
org $A9F2A8
    LDA #$0020           ; replaces: LDA #$0078

; Big Boy rising from Samus
org $A9F2BA 
    LDA #$0030           ; replaces: LDA #$00C0

; Big Boy backing off
org $A9F2E6
    LDA #$0016          ; replaces: LDA #$0058

; Big Boy going left guiltily
org $A9F31E 
    LDA #$0016          ; replaces: LDA #$0058 

; Big Boy going right guiltily
org $A9F34A
    LDA #$0016          ; replaces: LDA #$0100  

; Increase rate of draining Samus (without Varia):
org $A9C560
    LDY #$FFF9          ; replaces: LDY #$FFFC

; Increase rate of draining Samus (with Varia):
org $A9C569
    LDY #$FFF9          ; replaces: LDY #$FFFD

