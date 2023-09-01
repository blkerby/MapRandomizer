; Fast reload on death
; Based on patch by total: https://metroidconstruction.com/resource.php?id=421
; Compile with "asar" (https://github.com/RPGHacker/asar/releases)


!deathhook82 = $82DDC7 ;$82 used for death hook (game state $19)

;free space: make sure it doesnt override anything you have
!freespace82_start = $82FE00
!freespace82_end = $82FE80
!freespacea0 = $a0fe00 ;$A0 used for instant save reload

!QUICK_RELOAD = $1f60 ;dont need to touch this

lorom

incsrc "constants.asm"

; Hook Death Game event (19h)
org !deathhook82
deathhook:
    php
    rep #$30

    lda #$0001
    sta !QUICK_RELOAD ; Currently "quick reloading"

    jsl $82be17       ; Stop sounds
    jsl load_save_slot
	jsl $80858C		  ; load map

    ; In case we're on an elevator ride, reset this state so that Samus will have control after the reload:
    stz $0E18

    lda #$0006        
    sta $0998         ; Goto game mode 6 (load game)
    plp
    rts
    
warnpc $82DDF1

; Hook main game loop
org $82897A
    jsl hook_main

; $08, $14, $15, $16, $17

org !freespace82_start

hook_main:
    lda $0998
    cmp #$0007
    beq .check
    cmp #$0008
    beq .check
    cmp #$000c
    beq .check
    cmp #$0012
    beq .check
    cmp #$0013
    beq .check
    cmp #$0014
    beq .check
    cmp #$0015
    beq .check
    cmp #$0016
    beq .check
    cmp #$0017
    beq .check
    cmp #$0018
    beq .check
    ; inapplicable game state, so skip check for quick reload inputs.
    jsl $808338  ; run hi-jacked instruction
    rtl
.check:
    jsl $808338  ; run hi-jacked instruction
    jmp check_reload


check_reload:
    PHP
    REP #$30
    PHA

    ; Disable quick reload during the Samus fanfare at the start of the game (or when loading game from menu)
    lda $0A44
    cmp #$E86A
    beq .noreset

    lda $8B      ; Controller 1 input
    and #$3030   ; L + R + Select + Start
    cmp #$3030
    bne .noreset ; If any of the 4 inputs are not currently held, then do not reset.

    lda $8F      ; Newly pressed controller 1 input
    and #$3030   ; L + R + Select + Start
    bne .reset   ; Reset only if at least one of the 4 inputs is newly pressed
.noreset
    PLA
    PLP
    RTL
.reset:
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
    jml $80a0d9

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

    ; Stop music before starting new track. This prevents audio glitchiness in case the death track is playing.
    lda #$0000
    jsl $808FC1

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

; Determine which save slot to load from, and load it:
load_save_slot:
    lda $0E18       ; Check if we are on an elevator ride
    bne .current      ; If so, just load the current save (in spite of Samus facing forward, don't go back to previous save.)
    lda $0A1C       ; Check if Samus is still facing forward (initial state after loading)
    beq .forward     
    cmp #$009B
    beq .forward

    ; Load current save slot:
.current:
    ; if not reloading during death, then increment reload count
    lda $0998
    cmp #$0013
    bcs .skip_inc_stat
    lda !stat_reloads
    inc
    sta !stat_reloads
.skip_inc_stat:
    lda $0952
    jml $818085

.forward:
    ; Samus still facing forward, so we'll go back to previous save:
    lda $0952         ; Load saveslot
    beq .cycle
    dec               ; Decrease save slot by 1
    bra .check

.cycle:
    lda #$0002        ; Wrap back around to save slot 2

.check:
    sta $0952
    jsl $818085
    bcs .forward     ; If slot is empty/corrupt, go back to previous save again.

    lda !stat_loadbacks
    inc
    sta !stat_loadbacks
    rtl

warnpc $A18000