;;; Spring ball menu crash fix by strotlog.
;;; Fix obscure vanilla bug where: turning off spring ball while bouncing, can crash in $91:EA07,
;;; or exactly the same way as well in $91:F1FC.
;;; Adapted for map rando by Stag Shot:
;;; toggle-modifications nn_357.

arch snes.cpu
lorom

;;; these variable are defined by the crash_handle_base.asm patch and patch.rs

!crash_toggles = $85AD00 
!kill_samus = $85b5a0
!bug_dialog = $85b000

!bank_82_free_space2_start = $82f810 ; hook unpause (springball crash)
!bank_82_free_space2_end = $82f830

!bank_85_free_space_start = $85ad04
!bank_85_free_space_end = $85ad45

;;; vanilla hooks

org $8293bb
    jmp check_unpause

org $91ea07
    jsl spring_ball_crash

org $91f1fc
    jsl spring_ball_crash
    
;;; custom code


org !bank_82_free_space2_start
check_unpause:
    php
    sep #$20
    lda $00cf               ; pending crash ID
    stz $00cf
    cmp #$42                ; springball?
    bne .skip
    plp
    jmp $93c1               ; skip changing gamestate
.skip
    plp
    lda #$0008              ; replaced code
    jmp $93be

assert pc() <= !bank_82_free_space2_end


org !bank_85_free_space_start
spring_ball_crash:
    lda $0B20               ; morph bounce state
    cmp #$0600              ; bugged?
    bcc .skip
    lda !crash_toggles
    and #$000F
    beq .default
    cmp #$0002
    beq .fix
.warn
    lda #$0042
    jsl !bug_dialog
.fix
    lda #$0000
    stz $0b20
    rtl
.default
    sep #$20
    lda #$42                ; bug ID
    sta $00cf               ; set flag to prevent unpause from resetting gamestate to 8
    rep #$30
    jsl !bug_dialog
    jsl !kill_samus
    lda #$0000
    stz $0B20
    rtl
.skip
    lda $0B20               ; replaced code
    asl                     ;
    rtl

assert pc() <= !bank_85_free_space_end
