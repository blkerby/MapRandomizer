!bank_80_free_space_start = $80E3C0
!bank_80_free_space_end = $80E440

incsrc "constants.asm"

org $809BEE
    jsr hook_draw_tanks

org $82914F
    jsl hook_equipment_screen
    nop : nop

org !bank_80_free_space_start
hook_draw_tanks:
    lda !num_disabled_etanks
    beq .done
    sta $16
.loop:
    ldx $9CCE,y
    lda #$382F
    sta $7EC608,x
    iny
    iny
    dec $16
    bne .loop

.done:
    ; run hi-jacked instruction
    lda #$9DBF
    rts

hook_equipment_screen:
    lda $8B
    and #$2000  ; holding Select?
    beq .done

    lda $8F
    and #$0040  ; newly pressed X?
    bne .disable_tank
    
    lda $8F
    and #$4000  ; newly pressed Y?
    bne .enable_tank

.done:
    ; run hi-jacked instructions
    lda #$0001 
    sta $0763
    rtl

.disable_tank:
    lda $09C4
    cmp #$0065   ; Is max health >= 101?
    bmi .done

    ; Decrease max health by 100
    sec
    sbc #$0064
    sta $09C4

    ; Clamp current health to max health
    cmp $09C2
    bpl .skip_clamp
    sta $09C2

.skip_clamp:
    ; Increment disabled ETank count
    inc !num_disabled_etanks

    lda #$FFFF
    sta $0A06   ; set previous health to invalid value, to trigger it to be redrawn
    bra .done

.enable_tank:
    lda !num_disabled_etanks  ; is number of disabled ETanks non-zero?
    beq .done

    ; Decrement disabled ETank count
    dec !num_disabled_etanks

    ; Increase max health by 100
    lda $09C4
    clc
    adc #$0064
    sta $09C4

    lda #$FFFF
    sta $0A06   ; set previous health to invalid value, to trigger it to be redrawn

    bra .done
warnpc !bank_80_free_space_end

