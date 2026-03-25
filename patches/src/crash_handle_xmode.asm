;;; X-Mode collision fix - nn_357
;;; rewrite original input handler for solid tile collision to free up space.
;;; thanks to StagShot for noticing the rep$30 isn't needed here, it's already initialized by the single caller to this function.
;;; whole fix can now fit in the original function.

arch snes.cpu
lorom

incsrc "constants.asm"

;;; these variable are defined by the crash_handle_base.asm patch and patch.rs


!bank_91_free_space_start = $9195bc 
!bank_91_free_space_end =  $91965a 

org $91816f 
xmodefix:
    php
    jsr $81a9   ; this fixes regular xmode collision by using the correct pose lookup routine for all collisions
    lda $0a78   ; load frozen time variable
    beq .skip   ; if time is NOT frozen skip over the next instruction (jump to xmode crash handler.)
    lda #$0045  ; load the msg bug id for X-Mode Collision
    jsr xmode   ; now in same bank so uses a jsr.
.skip
    plp
    rts

assert pc() <= $918181  ; Make sure we don't overwrite the next routine.

;;;
;;; custom code - currently only using upto {9195D2} {approx 130bytes free, reserved for more future xmode warn fix.}
;;; 
org !bank_91_free_space_start
xmode:
    lda !crash_toggles
    and #$F000
    beq .default
    ;cmp #$0200
    ;beq .fix
.warn
    ;lda #$0045       ; crash dialog (warning) removed until better solution found, it will re-trigger many times until samus is out of collission so annoying.
    ;jsl !bug_dialog  ; there is space available here for additional code.
.fix
    rts
.default
    lda #$0045
    jsl !bug_dialog
    jsl !kill_samus
    rts

assert pc() <= !bank_91_free_space_end