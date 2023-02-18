; Fast reload on death
; Based on patch by total: https://metroidconstruction.com/resource.php?id=421
; Compile with "asar" (https://github.com/RPGHacker/asar/releases)


!deathhook82 = $82DDC7 ;$82 used for death hook (game state $19)

;free space: make sure it doesnt override anything you have
!freespace82_start = $82F800
!freespace82_end = $82F880
!freespacea0 = $a0fe00 ;$A0 used for instant save reload

!QUICK_RELOAD = $1f60 ;dont need to touch this

lorom


; Hook Death Game event (19h)
org !deathhook82
deathhook:
    php
    rep #$30
    lda #$0001
    sta !QUICK_RELOAD ; Currently "quick reloading"
    jsl $82be17       ; Stop sounds
	lda $0952         ; Load saveslot
    jsl $818085       ; Load savefile
	jsl $80858C		  ; load map
    lda #$0006        
    sta $0998         ; Goto game mode 6 (load game)
    plp
    rts

warnpc $82DDF1

; Hook main gameplay
org $828BB3
    JSL hook_main

; Hook state $14 (Samus ran out of health, black out surroundings)
org $82DD6B
    JSL hook_14

; Hook state $15 (Samus ran out of health, black out surroundings)
org $82DD74
    JSL hook_15

; Hook state $16 (Samus ran out of health, starting death animation)
org $82DD8A
    JSL hook_16

; Hook state $17 (Samus ran out of health, flashing)
org $82DD9D
    JSL hook_17

; Hook state $18 (Samus ran out of health, explosion)
org $82DDB2
    JSL hook_18

org !freespace82_start

hook_main:
    JSL $A09169  ; run hi-jacked instruction
    JMP check_reload

hook_14:
    JSL $808FF7  ; run hi-jacked instruction
    JMP check_reload

hook_15:
    JSL $908A00  ; run hi-jacked instruction
    JMP check_reload

hook_16:
    JSL $9BB43C  ; run hi-jacked instruction
    JMP check_reload

hook_17:
    JSL $9BB441  ; run hi-jacked instruction
    JMP check_reload

hook_18:
    JSL $9BB701  ; run hi-jacked instruction
    JMP check_reload

check_reload:
    PHP
    REP #$30
    PHA
    lda $8B      ; Controller 1 input
    and #$3030   ; L + R + Select + Start
    cmp #$3030
    beq reset
    PLA
    PLP
    RTL
reset:
    PLA
    PLP
    jsr deathhook
    RTL
warnpc !freespace82_end

; Hook setting up game
org $80a088
    jsl setup_music : nop : nop

org $80A095
    jml setup_game_1

org $80a0ce
    jml setup_game_2

org $80a113
    jml setup_game_3

org $91e164
    jsl setup_samus : nop : nop

; Free space somewhere for hooked code
org !freespacea0
setup_music:
    lda !QUICK_RELOAD
    bne .quick
    stz $07f3
    stz $07f5
.quick
    rtl

setup_game_1:
	jsl $82be17       ; Stop sounds
    lda !QUICK_RELOAD
    bne .quick
    lda #$ffff      ; Do regular things
    sta $05f5
    jml $80a09b
.quick
    jsl $80835d
    jsl $80985f
    jsl $82e76b
    jml $80a0aa

setup_game_2:
    jsl $82be17       ; Stop sounds
    lda !QUICK_RELOAD
    bne .quick
    jsl $82e071
    jml $80a0d2
.quick
    jml $80a0d5

setup_game_3:
    jsl $82be17       ; Stop sounds
    pha
    lda !QUICK_RELOAD
    bne .quick
    pla
    jsl $80982a
    jml $80a117
.quick
    pla
    jsl $80982a
    stz !QUICK_RELOAD
    lda $07c9
    cmp $07f5
    bne .loadmusic
    lda $07cb
    cmp $07f3
    bne .loadmusic
    jml $80a122

.loadmusic
    lda $07c9
    sta $07f5
    lda $07cb
    sta $07f3    

    lda $07cb
    ora #$ff00
    jsl $808fc1
    lda $07c9
    jsl $808fc1

    jml $80a122

setup_samus:
    lda !QUICK_RELOAD
    beq .normal
    lda #$e695
    sta $0a42
    lda #$e725
    sta $0a44
.normal    
    lda $09c2
    sta $0a12
    rtl

warnpc $A18000