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

!shown_warning = $7EF59C  ; whether a warning has already been shown since X-Ray initialized

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

org $91cafe
    jmp hook_setup_xray
return_setup_xray:

;;;
;;; custom code - currently only using up to {9195D2} {approx 130bytes free, reserved for more future xmode warn fix.}
;;; 
org !bank_91_free_space_start
xmode:
    lda !crash_toggles
    and #$F000
    beq .default
    ;cmp #$0200
    ;beq .fix
.warn
    lda !shown_warning
    bne .fix
    inc
    sta !shown_warning  ; set !shown_warning to 1
    lda #$0045
    jsl !bug_dialog
.fix
    rts
.default
    lda #$0045
    jsl !bug_dialog
    jsl !kill_samus
    rts

; initialize shown_warning to 0 when starting to use x-ray
hook_setup_xray:
    sta $0a78  ; run hi-jacked instruction: time-is-frozen flag = 1
    lda #$00
    sta !shown_warning
    jmp return_setup_xray

assert pc() <= !bank_91_free_space_end