;;; Fix auto-reserve / pause bug
;;;
;;; This patch will initiate the death sequence if pause hit with auto-reserve enabled
;;; on exact frame that leads to crash.
;;;
;;; (thanks to Benox50 for his initial patch, nn44/aquanight for the light pillar fix)
;;; 


;;; these variable are defined by the crash_handle_base.asm patch and patch.rs

!crash_toggles = $85AD00 
!kill_samus = $85b5a0
!bug_dialog = $85b000

!bank_82_free_space_start = $82fbf0 ; pause / reserve bug
!bank_82_free_space_end = $82fc30


;;; vanilla hooks

org $828b3f
    jsr pause_func : nop          ; gamestate 1Ch (unused) handler


;;; custom code
    
org !bank_82_free_space_start
pause_func:
    lda !crash_toggles
    and #$00F0
    beq .default
    cmp #$0020
    beq .fix
.warn
    lda #$0041                    ; "fix" leaves the screen black like any pause close to fadeout, warning will relight the screen due to showing the dialog
    jsl !bug_dialog
.fix
    lda #$0008
    sta $0998
    rts
.default
    lda #$0041                    ; msg ID
    jsl !bug_dialog
    jsl !kill_samus
    stz $9d6                      ; clear reserve health
    rts

assert pc() <= !bank_82_free_space_end
