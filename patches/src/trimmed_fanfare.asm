; Fanfare fix by Stag Shot / nn44
; 
; Fixes potential fanfare overlap with trimmed setting in rooms with two items.

lorom

!bank_84_free_space_start = $84F630
!bank_84_free_space_end = $84F68F

!ram_music_backup = $7EF59E

org $82e126
    jml check_music
    nop #3
    
org $848c01
    jsl clear_music

org $858491
    dw $00F0        ; 4s

org !bank_84_free_space_start
check_music:
    lda !ram_music_backup
    cmp #$ffff      ; music backup active?
    bne .normal_ret
    jml $82e134     ; skip rest of func
    
.normal_ret
    lda #$0000      ; replaced code
    jsl $808ff7     ;
    jml $82e12d
    
clear_music:
    phy
    pha             ; music track
    cmp #$0002
    bne .leave      ; fanfare?
    lda $7f5        ; current music
    cmp #$0002
    bne .leave_2    ; in middle of previous fanfare?
    lda #$0000
    jsl $808fc1     ; stop music (8-frame delay)
    pla
    jsl $808fc1     ; fanfare (8-frame delay)
    lda !ram_music_backup ; previous music
    ldy #$0168      ; 6s
    jsl $808ff7     ; music (delayed)
    lda #$ffff      ; clear music track
    sta !ram_music_backup
    bra .leave_3

.leave
    lda $7f5
.leave_2
    sta !ram_music_backup ; save music track
    pla             ; music track
    jsl $808fc1     ; replaced code
.leave_3
    ply
    rtl

assert pc() <= !bank_84_free_space_end
