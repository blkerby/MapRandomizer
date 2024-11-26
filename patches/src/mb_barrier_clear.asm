arch snes.cpu
lorom

!bank_84_freespace_start = $84F200
!bank_84_freespace_end = $84F300

org $83AAD2
    dw $EB00  ; Set door ASM for Rinka Room toward Mother Brain

macro check_objective(i,plm)
    lda.w #$007E
    sta.b $02
    lda.l ObjectiveAddrs+<i>
    sta.b $00
    lda.l ObjectiveBitmasks+<i>
    sta.b $04
    lda.b [$00]
    bit.b $04
    beq ?skip  ; skip clearing if objective not done

    jsl $8483D7
    db <plm>
    db $04
    dw clear_barrier_plm
?skip:
endmacro

; Free space in bank $8F
org $8FEB00
    ; clear barriers in mother brain room based on main bosses killed:
    %check_objective(0,$39)
    %check_objective(2,$38)
    %check_objective(4,$37)
    %check_objective(6,$36)

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

warnpc $8FEBC0
org $8FEBC0
ObjectiveAddrs:
    dw $D829, $D82A, $D82B, $D82C
ObjectiveBitmasks:
    dw $0001, $0001, $0001, $0001

; OBJECTIVE: None (must match address in patch.rs)
warnpc $8FECA0
org $8FECA0

    jsl $8483D7
    db $39
    db $04
    dw clear_barrier_plm

    jsl $8483D7
    db $38
    db $04
    dw clear_barrier_plm

    jsl $8483D7
    db $37
    db $04
    dw clear_barrier_plm

    jsl $8483D7
    db $36
    db $04
    dw clear_barrier_plm

    jmp motherbrain

warnpc $8FED00


; Remove invisible spikes where Mother Brain used to be (common routine used by both the left and right door ASMs)
org !bank_84_freespace_start

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
    lda $7ed823  ; set flag to represent that Bowling statue has been used (flag bit unused by vanilla game)
    ora #$0001
    sta $7ed823
    rtl

pirates_set_flag:
    jsl $8081FA  ; run hi-jacked instruction (set zebes awake flag)

    lda $079B   ; room pointer    
    cmp #$975C  ; pit room?
    bne .not_pit_room
    lda $7ED823
    ora #$0002
    sta $7ED823
    rtl
.not_pit_room:
    cmp #$A521  ; baby kraid room?
    bne .not_baby_kraid_room
    lda $7ED823
    ora #$0004
    sta $7ED823
    rtl
.not_baby_kraid_room:
    cmp #$D2AA  ; plasma room?
    bne .not_plasma_room
    lda $7ED823
    ora #$0008
    sta $7ED823
    rtl
.not_plasma_room:
    cmp #$B62B  ; metal pirates room?
    bne .not_metal_pirates_room
    lda $7ED823
    ora #$0010
    sta $7ED823
.not_metal_pirates_room:
    rtl

warnpc !bank_84_freespace_end

; bowling chozo hook
org $84D66B
    jsl bowling_chozo_set_flag

; enemies (pirates) dead hook
org $84BE0E
    jsl pirates_set_flag
