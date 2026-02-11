!bank_90_free_space_start = $90FC40
!bank_90_free_space_end = $90FCC0

!bank_91_free_space_start = $91F7F4
!bank_91_free_space_end = $91F88C

!equipped_items = $09A2
!fake_shinecharge = $7EF597

; bitmask for equipped items ($09A2) and collected items ($09A4):
!blue_booster = $0040
!spark_booster = $0080
!speed_booster = $2000
!any_booster = !blue_booster|!spark_booster|!speed_booster

; Accelerate Samus' animation with any booster item:
org $908502
    bit #!any_booster

; Increase Samus' dash counter with any booster item:
org $90855B
    bit #!any_booster

; Raise dash speed cap from $2.0 to $7.0 with any booster item:
org $90978C
    bit #!any_booster

; When only Spark Booster is equipped, lose speed when jumping or falling:
; This caps Samus' dash speed to $2.0, resets dash counter to zero, and remove echoes.
org $909919 : jsr hook_jump
org $91E901 : jsr hook_fall

; Apply speedy jump height with Blue Booster or Speed Booster:
org $90991C : bit #!blue_booster|!speed_booster   ; regular jump
org $9099A9 : bit #!blue_booster|!speed_booster   ; wall jump

; Graphical change to not show Samus as blue when running with Spark Booster:
org $91D9EA
    jsr hook_speed_boosting
    
; Graphical change to show fake shinecharge palette:
org $91DAE3
    jsr hook_shinecharge_palette

; Prevent shinespark from happening with fake shinecharge, and reset dash counter 
; (to ensure you still have to wait out the fake shinecharge for temp blue):
org $91F564 : jsr hook_normal_jump_spark
org $91F571 : jsr hook_normal_jump_spark

; When running with only Blue Booster is equipped, replace shinecharge with "fake shinecharge".
; This flashes blue instead of white and does not allow a shinespark.
; Shinecharging using a blue suit still gives a real shinecharge.
org $91F7C1
    jsr hook_shinecharge

; When only Blue Booster is equipped, disable speed echoes when having dash speed:
org $90EEE7
    jsr hook_update_speed_echoes

org !bank_91_free_space_start


hook_speed_boosting:
    ; If case of any item combination other than exactly Spark Booster, behave like vanilla:
    lda !equipped_items
    and #!any_booster
    cmp #!spark_booster
    bne .vanilla

    ; If Samus does not have running momentum, behave like vanilla (e.g. to be able to show blue suit):
    lda $0B3C  ; Samus running momentum flag
    beq .vanilla

    ; If Samus is running with Spark Booster, don't show blue:
    lda #$0000
    rts

.vanilla:
    ; Vanilla behavior: show blue palette if dash counter = 4
    lda $0B3E
    rts

hook_shinecharge_palette:
    ; load vanilla palette
    lda !fake_shinecharge
    beq .vanilla
    lda fake_shinecharge_palette_table,x
    rts
.vanilla:
    lda $DB10,x
    rts

hook_shinecharge:
    lda !equipped_items
    and #!any_booster
    cmp #!blue_booster
    bne .vanilla

    ; If Samus does not have running momentum, behave like vanilla;
    ; e.g. to be able to shinecharge with blue suit:
    lda $0B3C  ; Samus running momentum flag
    beq .vanilla

    lda #$0001  ; fake shinecharge
    bra .done
.vanilla:
    lda #$0000  ; regular shinecharge
.done:
    sta !fake_shinecharge
    lda #$0001  ; run hi-jacked instruction
    rts

fake_shinecharge_palette_table:
    dw .no_suit, .varia_suit, .gravity_suit
.no_suit:
    dw $9B20, $9B40, $9B60, $9B80, $9B60, $9B40
.varia_suit:
    dw $9D20, $9D40, $9D60, $9D80, $9D60, $9D40
.gravity_suit:
    dw $9FA0, $9F20, $9F40, $9F60, $9F40, $9F20

hook_normal_jump_spark:
    lda !fake_shinecharge
    beq .skip

    ; If attempting to activate a "fake shinecharge", clear the dash counter.
    ; This makes it so you still have to wait out the fake shinecharge when doing temporary blue,
    ; or the blue is lost.
    lda #$0000
    sta $0B3E
    rts
.skip:
    lda $0A68   ; run hi-jacked instruction
    rts


hook_fall:
    jsl spark_booster_lose_speed
    lda $0a1e  ; run hi-jacked instruction
    rts

assert pc() <= !bank_91_free_space_end

org !bank_90_free_space_start

spark_booster_lose_speed:
    ; If either Blue Booster or Speed Booster is equipped, skip the speed clamp/loss.
    lda !equipped_items
    bit #!blue_booster|!speed_booster
    bne .skip_speed_clamp

    ; If Samus does not have running momentum, skip the speed clamp/loss (to avoid interfering with blue suit).
    lda $0B3C  ; Samus running momentum flag
    beq .skip_speed_clamp

    ; Set speed echoes to merge back into Samus
    lda #$FFFF
    sta $0AAE

    stz $0b3e   ; Clear dash counter

    ; Clamp extra run speed to $2.0:
    lda $0b42
    cmp #$0002
    bcc .skip_speed_clamp
    lda #$0002
    sta $0b42
    stz $0b44

    lda #$0000
    rtl
.skip_speed_clamp:
    lda $09a2   ; run hi-jacked instruction (relevant only where called by hook_jump)
    rtl

hook_jump:
    jsl spark_booster_lose_speed
    rts

hook_update_speed_echoes:
    lda !equipped_items
    and #!any_booster
    cmp #!blue_booster
    bne .vanilla

    ; If Samus does not have running momentum, show echoes as normal (so they can appear with blue suit).
    lda $0B3C  ; Samus running momentum flag
    beq .vanilla

.hide_echoes
    lda #$0000
    rts
.vanilla:
    lda $0b3e  ; run hi-jacked instruction
    rts

assert pc() <= !bank_90_free_space_end
