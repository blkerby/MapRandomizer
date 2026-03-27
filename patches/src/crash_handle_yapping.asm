; Yapping maw shinespark crash
; Noted by PJBoy: https://patrickjohnston.org/bank/A8#fA68A


arch snes.cpu
lorom

incsrc "constants.asm"

;;; these variable are defined by the crash_handle_base.asm patch and patch.rs


!bank_85_free_space_start = $85b5b4
!bank_85_free_space_end = $85b5f1

!bank_90_free_space_start = $90fc10
!bank_90_free_space_end = $90fc20


;;; hooks into vanilla code

org $90d354
    jsr yapping_maw_crash
    
;;; custom code

org !bank_90_free_space_start
yapping_maw_crash:
    cmp #$0005              ; valid table entries are 0-2 * 2
    bcc .skip
    jsl ym_crash
    rts
.skip
    jmp ($d37d,x)           ; valid entry
    
assert pc() <= !bank_90_free_space_end

org !bank_85_free_space_start
ym_crash:
    lda !crash_toggles
    and #$0f00
    beq .default
    cmp #$0200
    beq .fix
.warn
    lda #$0043              ; bug ID
    jsl !bug_dialog
.fix
    lda #$d3f3
    sta $0a58
    lda #$001e
    sta $0aa2
    sta $0ac0
    sta $0ac2
    lda #$0000
    ldx #$0000
    ldy #$0004
    rtl
.default
    lda #$0043              ; bug ID
    jsl !bug_dialog
    jsl !kill_samus
    rtl
    

assert pc() <= !bank_85_free_space_end