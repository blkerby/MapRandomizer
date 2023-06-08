arch snes.cpu
lorom

org $83AAD2
    dw $EB00  ; Set door ASM for Rinka Room toward Mother Brain (using Bosses as default objective)

; Free space in bank $8F 

; OBJECTIVE: Bosses (must match address in patch.rs)
org $8FEB00
    ; clear barriers in mother brain room based on main bosses killed:
kraid:
    lda $7ed829
    bit #$0001
    beq .skip  ; skip clearing if kraid isn't dead

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm
.skip:

phantoon:
    lda $7ed82b
    and #$0001
    beq .skip  ; skip clearing if phantoon isn't dead

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm
.skip:

draygon:
    lda $7ed82c
    bit #$0001
    beq .skip  ; skip clearing if draygon isn't dead

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm
.skip:

ridley:
    lda $7ed82a
    bit #$0001
    beq .skip  ; skip clearing if ridley isn't dead

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm
.skip:

motherbrain:
    lda $7ed82d
    bit #$0001
    beq done  ; skip clearing if mother brain isn't dead

    ; Spawn Mother Brain's room escape door:
    jsl $8483D7
    dw  $0600,  $B677

    ; Remove invisible spikes where Mother Brain used to be:
    jsl remove_spikes
done:
    rts

; OBJECTIVE: Minibosses (must match address in patch.rs)
warnpc $8FEB60
org $8FEB60
    ; clear barriers in mother brain room based on mini-bosses killed:
spore_spawn:
    lda $7ed829
    bit #$0002
    beq .skip  ; skip clearing if spore spawn isn't dead

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm
.skip:

crocomire:
    lda $7ed82a
    and #$0002
    beq .skip  ; skip clearing if crocomire isn't dead

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm
.skip:

botwoon:
    lda $7ed82c
    bit #$0002
    beq .skip  ; skip clearing if draygon isn't dead

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm
.skip:

golden_torizo:
    lda $7ed82a
    bit #$0004
    beq .skip  ; skip clearing if GT isn't dead

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm
.skip:

    jmp motherbrain


; OBJECTIVE: Metroids (must match address in patch.rs)
warnpc $8FEBC0
org $8FEBC0
    ; clear barriers in mother brain room based on Metroid rooms cleared:
metroid_1:
    lda $7ed822
    bit #$0001
    beq .skip

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm
.skip:

metroid_2:
    lda $7ed822
    and #$0002
    beq .skip

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm
.skip:

metroid_3:
    lda $7ed822
    bit #$0004
    beq .skip

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm
.skip:

metroid_4:
    lda $7ed822
    bit #$0008
    beq .skip

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm
.skip:

    jmp motherbrain


; OBJECTIVE: Chozos (must match address in patch.rs)
warnpc $8FEC20
org $8FEC20
    ; clear barriers in mother brain room based on Chozos defeated/activated:

bomb_torizo:
    lda $7ed828
    bit #$0004
    beq .skip  ; skip clearing if Bomb Torizo isn't dead

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm
.skip:

bowling_chozo:
    lda $7ed822
    and #$0010
    beq .skip  ; skip clearing if Bowling Alley Chozo hasn't been used

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm
.skip:

acid_chozo:
    lda $7ed821
    bit #$0010
    beq .skip  ; skip clearing if Acid Chozo hasn't been used

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm
.skip:

golden_torizo_chozo:
    lda $7ed82a
    bit #$0004
    beq .skip  ; skip clearing if Golden Torizo isn't dead

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm
.skip:

    jmp motherbrain

warnpc $8FED00

org $83AAEA
    dw $EE00  ; Set door ASM for Tourian Escape Room 1 toward Mother Brain

org $83AAE3
    db $00    ; Set door direction = $00  (to make door not close behind Samus)

; Custom door ASM for Tourian Escape Room 1 toward Mother Brain
org $8FEE00
    jsl $8483D7            ;\
    db  $00, $06           ;|
    dw  $B677              ;} Spawn Mother Brain's room escape door

    ; Remove invisible spikes where Mother Brain used to be:
    jsl remove_spikes

    rts

; Remove invisible spikes where Mother Brain used to be (common routine used by both the left and right door ASMs)
org $84F200
remove_spikes:
    ; Remove invisible spikes
    lda #$8000   ; solid tile
    ldx #$0192   ; offset to spike above Mother Brain right
    jsr $82B4
    lda #$8000   ; solid tile
    ldx #$0210   ; offset to spike above Mother Brain center-right
    jsr $82B4
    lda #$8000   ; solid tile
    ldx #$0494   ; offset to spike below Mother Brain right
    jsr $82B4
    rtl

clear_barrier_plm:
    dw $B3D0, clear_barrier_inst

clear_barrier_inst:
    dw $0001, clear_barrier_draw
    dw $86BC

clear_barrier_draw:
    dw $8006, $00FF, $00FF, $00FF, $00FF, $00FF, $00FF, $0000

bowling_chozo_set_flag:
    jsl $90F084  ; run hi-jacked instruction
    lda $7ed822  ; set flag to represent that Bowling statue has been used (flag bit unused by vanilla game)
    ora #$0010
    sta $7ed822
    rtl

warnpc $84F280

; bowling chozo hook
org $84D66B
    jsl bowling_chozo_set_flag

