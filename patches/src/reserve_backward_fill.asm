!bank_85_free_space_start = $85A050
!bank_85_free_space_end = $85A100

; Hook code that run when equipment screen tanks mode (Auto/Manual) is selected:
org $82AE8E
    jsl hook_reserve_refill
    nop

; update reserve refill code to set the counter value to maximum (#$FFFF) rather than
; to the current amount in reserves. This prevents the refill from abruptly stopping
; if switching back and forth between forward and reverse fill.
org $82AF5E
    lda #$FFFF
    sta $0757
    bra process_reserve_counter

warnpc $82AF6B
org $82AF6B
process_reserve_counter:

org !bank_85_free_space_start
hook_reserve_refill:
    lda $09C0
    cmp #$0002  ; reserves on manual?
    bne .done

    lda $8B
    and #$8800
    cmp #$8800  ; holding up + B?
    bne .done

    ; skip if samus energy <= 1
    lda $09C2
    cmp #$0002
    bmi .done

    ; play sound every 8 frames:
    LDA $0757
    DEC A
    STA $0757
    AND #$0007
    BNE .refill
    LDA #$002D
    JSL $80914D    ;} Queue sound 2Dh, sound library 3, max queued sounds allowed = 6 (gaining/losing incremental health)    

.refill:
    ; do reverse refill, where energy transfers into reserves:
    lda $09C2
    dec
    sta $09C2

    lda $09D6
    cmp $09D4
    bpl .overfull
    inc
    sta $09D6
    bra .done

.overfull:
    ; reserves are being overfilled, so set regular energy to 1
    ; (mirroring vanilla quirky behavior that sets reserve energy to 0 when overfilling regular energy)
    lda #$0001
    sta $09C2

.done:
    ; run hi-jacked code
    lda $8f
    bit #$0080
    rtl

warnpc !bank_85_free_space_end