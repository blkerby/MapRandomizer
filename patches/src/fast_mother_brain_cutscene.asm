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

; End phase 2 sooner:
; Reorder the timer check and the Samus health check, so if Samus health is low enough already then it doesn't wait for the timer.
; (We don't eliminate the timer delay entirely as that could interfere with execution of the stand-up glitch.)
org $A9BB06
mb_phase_2_end_decide:
    LDA $09C2
    CMP #$0190             ;} If [Samus health] >= 400:
    BMI .finish_samus      ;/
    DEC $0FB2              ; Decrement Mother Brain's body function timer
    BPL .done              ; If [Mother Brain's body function timer] >= 0: return
    LDA #$B8EB
    STA $0FA8              ;} Mother Brain's body function = $B8EB (firing rainbow beam)
.done:
    RTS                    ; Return
warnpc $A9BB1A
org $A9BB1A
.finish_samus:

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