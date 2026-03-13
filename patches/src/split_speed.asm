!bank_90_free_space_start = $90FC40
!bank_90_free_space_end = $90FD00

!bank_91_free_space_start = $91F7F4
!bank_91_free_space_end = $91F88C

!bank_91_free_space_start2 = $91FC42
!bank_91_free_space_end2 = $91FC66

!bank_82_free_space_start = $82C42D
!bank_82_free_space_end =   $82C465

!bank_82_freespace2_start = $82FF20
!bank_82_freespace2_end   = $82FF30

!bank_B6_free_space_start = $B6FC00
!bank_B6_free_space_end = $B6FE00

!equipped_items = $09A2
!fake_shinecharge = $7EF597

; bitmask for equipped items ($09A2) and collected items ($09A4):
!blue_booster = $0040
!spark_booster = $0080
!speed_booster = $2000
!any_booster = !blue_booster|!spark_booster|!speed_booster

org $82A22F
    lda #list_boots_ram_tilemaps

org $82A23D
    lda list_boots_equip_bitmasks, y

org $82A257
    ldx list_boots_equip_tilemaps, y

org $82A25D
    lda list_boots_equip_bitmasks, y

org $82B1C6
    cmp #$0303

org $82B517
    bit list_boots_equip_bitmasks, x

org $82B545
    bit list_boots_equip_bitmasks, x

org $82B51E
    cpx #$0008

org $82C194
    dw boots_item_selector_positions

org $82A278
    cpy #$0008

org $82C032
    dw list_boots_ram_tilemaps

org $82C03A
    dw list_boots_equip_bitmasks

org $82C04A
    dw list_boots_equip_tilemaps
    
org $82C2A6
list_boots_equip_tilemaps:
    dw $BFD2, $BFE4, tilemap_blue_booster, tilemap_spark_booster
warnpc $82C2B7

org !bank_82_free_space_start

;Tilemaps
;oBLUE BOOSTER
tilemap_blue_booster:
    dw $12FF, $1270, $1271, $1272, $1273, $1274, $1275, $1276, $1277

;oSPARK BOOSTER
tilemap_spark_booster:
    dw $12FF, $1278, $1279, $127A, $127B, $127C, $127D, $127E, $127F

; Replacement boots table
list_boots_equip_bitmasks:
    dw $0100, $0200, !blue_booster, !spark_booster

list_boots_ram_tilemaps:
    dw $3CEA, $3D2A, $3D6A, $3DAA

warnpc !bank_82_free_space_end
assert pc() <= !bank_82_free_space_end

org !bank_82_freespace2_start

boots_item_selector_positions:
    dw $00CC, $009C
    dw $00CC, $00A4
    dw $00CC, $00AC
    dw $00CC, $00B4

warnpc !bank_82_freespace2_end
assert pc() <= !bank_82_freespace2_end

; Accelerate Samus' animation with any booster item:
org $908502
    bit #!any_booster

; Increase Samus' dash counter with any booster item:
org $90855B
    bit #!any_booster

; Initialize speed booster running with any booster item:
org $90978C
    bit #!any_booster

; Clamp dash speed to $2.0 when only Spark Booster is equipped
org $9097A9
    jmp hook_speed_clamp

; When only Spark Booster is equipped, lose speed when jumping or falling:
; This resets dash counter to zero and remove echoes.
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

; Secondary effects of not having speed items:

org $91E647
    ; Require Blue Booster or Speed Booster to retain bluesuit when unpausing
    jml hook_blue_unpause
    nop

org $84B7F2
    ; Lavaquake starts if any Speed Booster or Blue Booster is collected
    and #!blue_booster|!speed_booster

org $91DB0B
    jsr hook_reset_special_palette ; Palette reset from shinecharge special palette
org $91DB6D
    jsr hook_reset_special_palette ; Palette reset from shinespark
org $91DBF2
    jsr hook_reset_special_palette ; Palette reset from Crystal Flash shutdown
org $91DD23
    jsr hook_reset_special_palette ; Palette reset from X-Ray setup
    
org $91D9CD
    jmp hook_special_palette_fx_liquid_check
    nop

org $91D9D6
    jmp hook_special_palette_lavaacid_check

org !bank_91_free_space_start

hook_reset_special_palette:
    ;Hijacked instruction: Special Samus palette type = 0 (screw attack / speedbooster)
    stz $0ACC
    ; Clear the fake shinecharge state
    lda #$0000
    sta !fake_shinecharge
    rts

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
    jsl spark_booster_lose_blue
    lda $0a1e  ; run hi-jacked instruction
    rts
    

assert pc() <= !bank_91_free_space_end

org !bank_91_free_space_start2
hook_special_palette_fx_liquid_check:
    ; Hijacked code
    bne hook_special_palette_notliquid
    bra hook_special_palette_bluebooster

hook_special_palette_lavaacid_check:
    cmp $14
    bmi hook_special_palette_bluebooster
hook_special_palette_notliquid:
    jmp $D9DA

hook_special_palette_bluebooster:
    lda !equipped_items
    and #!any_booster
    cmp #!blue_booster
    bne .vanilla
    lda $0B3C   ;running momentum flag
    beq .vanilla    ; Vanilla bluesuit effect
    jmp $D9EA ;BRANCH_NOT_SCREW_ATTACKING
.vanilla
    jmp $D9F5 ;Return carry set

assert pc() <= !bank_91_free_space_end2

org !bank_90_free_space_start

hook_blue_unpause:
    bit #!speed_booster|!blue_booster
    bne .keep_blue
    
    bit #!spark_booster
    beq .no_blue
    
    lda $0B3C ; Running momentum flag
    beq .no_blue

    lda $0A1F
    and #$00ff
    cmp #$0001 ; Samus movement type = running - keep spark booster.
    beq .keep_blue
    
.no_blue
    jml $91E64C

.keep_blue
    jml $91E66F


spark_booster_lose_blue:
    ; If either Blue Booster or Speed Booster is equipped, skip the speed clamp/loss.
    lda !equipped_items
    bit #!blue_booster|!speed_booster
    bne .skip_lose_blue

    ; If Samus does not have running momentum, skip the speed clamp/loss (to avoid interfering with blue suit).
    lda $0B3C  ; Samus running momentum flag
    beq .skip_lose_blue

    ; Set speed echoes to merge back into Samus
    ;lda #$FFFF
    ;sta $0AAE
    jsl .remove_echoes

    stz $0b3e   ; Clear dash counter

    lda #$0000
    rtl
.skip_lose_blue:
    lda $09a2   ; run hi-jacked instruction (relevant only where called by hook_jump)
    rtl
.remove_echoes
    php
    phb
    jml $91DE8D

hook_jump:
    jsl spark_booster_lose_blue
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

hook_speed_clamp:
    lda !equipped_items
    and #!any_booster
    cmp #!spark_booster
    beq .spark_booster

    lda $0B42  ; run hi-jacked instruction
    jmp $97AC  ; jump to vanilla code to clamp dash speed to $7.0
.spark_booster
    jmp $97D5  ; jump to vanilla code to clamp dash speed to $2.0

assert pc() <= !bank_90_free_space_end

org $B6CE00
menu_tiles_blue_booster:
    db $FF, $FF, $FF, $FF, $8D, $FF, $B5, $FF, $8D, $FF, $B5, $FF, $8C, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $DA, $FF, $DA, $FF, $DA, $FF, $DA, $FF, $66, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $1C, $FF, $FD, $FF, $3C, $FF, $FD, $FF, $1C, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $73, $FF, $AD, $FF, $6D, $FF, $AD, $FF, $73, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $9C, $FF, $6B, $FF, $68, $FF, $6F, $FF, $98, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $44, $FF, $ED, $FF, $6C, $FF, $6D, $FF, $EC, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $23, $FF, $ED, $FF, $63, $FF, $ED, $FF, $2D, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
.end

menu_tiles_spark_booster:
    db $FF, $FF, $FF, $FF, $C4, $FF, $BD, $FF, $84, $FF, $F5, $FF, $8D, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $73, $FF, $AD, $FF, $6D, $FF, $E1, $FF, $ED, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $1B, $FF, $6A, $FF, $19, $FF, $6A, $FF, $6B, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $71, $FF, $F6, $FF, $F1, $FF, $F6, $FF, $71, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $CE, $FF, $B5, $FF, $B5, $FF, $B5, $FF, $CE, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $71, $FF, $AF, $FF, $A1, $FF, $BD, $FF, $63, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $10, $FF, $B7, $FF, $B1, $FF, $B7, $FF, $B0, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
    db $FF, $FF, $FF, $FF, $8F, $FF, $B7, $FF, $8F, $FF, $B7, $FF, $B7, $FF, $FF, $FF
    db $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF, $00, $FF
.end

org $B6
; Replace the SPEED BOOSTER row with BLUE/SPARK BOOSTER and move the bottom of the box down one row.
org $B6ED68
    dw $F954, $0CFF, $0D70, $0D71, $0D72, $0D73, $0D74, $0D75, $0D76, $0D77, $7940
org $B6EDA8
    dw $3940, $0CFF, $0D78, $0D79, $0D7A, $0D7B, $0D7C, $0D7D, $0D7E, $0D7F, $7940
org $B6EDE8
    dw $B941, $B942, $B942, $B942, $B942, $B942, $B942, $B942, $B942, $B942, $F941