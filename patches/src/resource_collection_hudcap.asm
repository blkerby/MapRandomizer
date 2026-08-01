!bank_84_free_space_start = $84F4A0
!bank_84_free_space_end = $84F4FF

org $84896C
    jsr add_max_energy_capped

org $84898A
    jsr add_max_reserve_capped

org $8489AD
    jsr add_max_missile_capped

org $8489B7
    jsr add_current_missile_capped

org $8489D6
    jsr add_max_super_capped

org $8489E0
    jsr add_current_super_capped

org $8489FF
    jsr add_max_powerbomb_capped

org $848A09
    jsr add_current_powerbomb_capped

org !bank_84_free_space_start

add_max_energy_capped:
    adc $0000,Y
    cmp #$05DB ;1499 energy
    bcc +
    lda #$05DB
+
    rts

add_max_reserve_capped:
    adc $0000,Y
    cmp #$0190
    bcc +
    lda #$0190
+
    rts

add_max_missile_capped:
    adc $0000,Y
    cmp #$03E7  ;999 missiles
    bcc +
    lda #$03E7
+
    rts

add_max_super_capped:
add_max_powerbomb_capped:
    adc $0000,Y
    cmp #$0063  ;99 supers/powerbombs
    bcc +
    lda #$0063
+
    rts

add_current_missile_capped:
    adc $0000,Y
    cmp $09C8
    bcc +
    lda $09C8
+
    rts

add_current_super_capped:
    adc $0000,Y
    cmp $09CC
    bcc +
    lda $09CC
+
    rts

add_current_powerbomb_capped:
    adc $0000,Y
    cmp $09D0
    bcc +
    lda $09D0
+
    rts

 print pc
assert pc() <= !bank_84_free_space_end