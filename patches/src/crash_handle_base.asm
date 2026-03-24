;;; crash handler (displays a custom message box explaining the crash)
;;; StagShot, toggle modifications - nn_357
;;;
;;; Implementation of custom dialog boxes
;;; Requires hooking multiple functions to support extended msg IDs (0x40+)
;;; and additional lookup tables
;;; 
;;; This code is used by other patches that handle hardlocks caused by the unequip springball, frame perfect pause on autoreserve trigger, yappingmaw shinespark
;;; X-mode solid tile collision.

arch snes.cpu
lorom

!bank_85_free_space_start = $85b000 ; be careful if moving this, the subpatches that use this code assume this location (xmode/oob/yappingmaw/autoreserve/springball)
!bank_85_free_space_end = $85b5a0

!bank_85_free_space2_start = $85b5a0 ; kill samus routine, location called by sub patches, as above, be careful if ever moving this code and update subpatches.
!bank_85_free_space2_end = $85b5b4

!msg_crash_timer_override = $7EF596 ; temporary variable used for overriding messagebox close delay times during crash box.
!crash_toggles = $85ad00  ; this location is looked at by the subpatches to decide if to call the handler etc.

org !crash_toggles
    dw $0000    ; overwritten by patch.rs if any values are changed, subpatches reference this. default is gameover / death (pre-toggleable setting)

;;; hooks into vanilla code

org $858493     ; override messagebox delay if crash dialog 
    jsr hook_msgbox_delay

org $858093
    jsr hook_message_box
    
org $8582e5
    jsr hook_index_lookup

org $8582ee
    jsr hook_message_table

org $85840c     ; hook unpause to prevent resetting gamestate to 8 if crash ID set
    jsr hook_button_lookup

;;; custom code

org !bank_85_free_space_start
bug_dialog:                 ; A = msg ID
    and #$00ff
    pha
    sep #$20
    lda #$0f                ; restore screen brightness to full
    sta $51
    sta !msg_crash_timer_override   ; messagebox timer will check if this is 0 (if its non zero load a longer time)
    rep #$30
    jsl $808338             ; wait for NMI

    pla                     ; dlg box parameter
    jsl $858080             ; dlg box
    cmp #$0044
    bne .skipkill           ; oob death (dlg 44) is removable via major glitches patch, if its thrown then the intent is to kill as it isn't toggleable.
    jsl kill_samus
.skipkill
    rtl
    
hook_message_box:
    rep #$30
    lda $1c1f
    cmp #$0040              ; custom boxes >= 0x40
    bcs .custom
    jmp $8241               ; original func
    
.custom
    ldx #(new_message_boxes-$869b) ; ptr for extended lookup table
    jmp $824f

hook_index_lookup:
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts

.custom
    sec
    sbc #$0040
    rts

hook_message_table:
    adc $34                         ; replaced code
    tax                             ;
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts
    
.custom
    txa
    clc
    adc #(new_message_boxes-$869b)  ; adjust ptr for extended table
    tax
    rts

hook_button_lookup:
    lda $1c1f
    cmp #$0040
    bcs .custom
    rts
    
.custom
    lda #$0001                      ; blank button tilemap
    ldy #(reserve_pause_msg-$8426)  ; blank button letter
    rts
    
hook_msgbox_delay: 
    pha
    lda !msg_crash_timer_override
    beq .nochange
    ldx #$005a ; (put 1.5 seconds on the clock) 
    lda #$00
    sta !msg_crash_timer_override ; clear our special variable so then next msgbox will have whatever timer was set on generation
.nochange
    pla
    jsr $8136  ; hi-jacked instruction.
    rts

; custom messages start at 0x41
new_message_boxes:
    dw $83c5, $825a, reserve_pause_msg  ; 0x41
    dw $83c5, $825a, springball_msg     ; 0x42
    dw $83c5, $825a, yapping_maw_msg    ; 0x43
    dw $83c5, $825a, oob_msg            ; 0x44
    dw $83c5, $825a, xmode_msg          ; 0x45
    dw $0000, $0000, msg_end

table "tables/dialog_chars.tbl",RTL

reserve_pause_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   PAUSED ON EXACT FRAME  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   AUTO-REFILL STARTED!   ", $000e,$000e,$000e

springball_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "  UNEQUIPPED SPRING BALL  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   IN NEUTRAL BOUNCE!     ", $000e,$000e,$000e
    
yapping_maw_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "  YAPPING MAW SHINESPARK  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, " END WITH NO INPUTS HELD! ", $000e,$000e,$000e

oob_msg:
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   SAMUS OUT-OF-BOUNDS!   ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    
xmode_msg:
    dw $000e,$000e,$000e, "        GAME CRASH!       ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "                          ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "   X-MODE TILE COLLISION  ", $000e,$000e,$000e
    dw $000e,$000e,$000e, "COLLIDED WITH A SOLID TILE", $000e,$000e,$000e
    
msg_end:

assert pc() <= !bank_85_free_space_end

org !bank_85_free_space2_start
kill_samus:
    lda #$8000            ; init death sequence (copied from $82db80)
    sta $a78
    lda #$0011
    jsl $90f084
    lda #$0013              ; set gamestate
    sta $998
    rtl

assert pc() <= !bank_85_free_space2_end